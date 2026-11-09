"""Парное сравнение четырёх адаптивных тестов на синтетических данных."""

import argparse
import csv
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


TOPICS = ("Информационная безопасность", "Охрана труда", "Документооборот")
TOPIC_ORDER = tuple(range(len(TOPICS))) * 4
LEVELS = ("лёгкий", "средний", "сложный")
THRESHOLD = 0.0
METHODS = {"rules": "По правилам", "mst": "MST", "cat": "CAT 2PL", "cat_3pl": "CAT 3PL"}


@dataclass(frozen=True)
class Item:
    index: int
    topic: int
    level: int
    a: float
    b: float
    c: float = 0.0

    def probability(self, theta: float, *, use_guessing: bool = False) -> float:
        p = 1 / (1 + math.exp(-self.a * (theta - self.b)))
        if use_guessing:
            return self.c + (1 - self.c) * p
        return p

    def information(self, theta: float, *, use_guessing: bool = False) -> float:
        p = self.probability(theta, use_guessing=use_guessing)
        if use_guessing:
            logistic = (p - self.c) / (1 - self.c)
            return self.a**2 * logistic**2 * (1 - p) / p
        return self.a**2 * p * (1 - p)


Trace = list[tuple[Item, int]]
Ask = Callable[[Item], int]
Pools = dict[tuple[int, int], list[Item]]


def make_bank(rng: random.Random, guessing_rng: random.Random) -> tuple[list[Item], Pools]:
    bank = []
    pools = {}
    for topic in range(len(TOPICS)):
        for level, center in enumerate((-1.5, 0.0, 1.5)):
            pool = []
            for _ in range(10):
                item = Item(
                    index=len(bank), topic=topic, level=level,
                    a=rng.uniform(0.8, 1.8), b=rng.uniform(center - 0.5, center + 0.5),
                    c=guessing_rng.uniform(0.10, 0.25),
                )
                pool.append(item)
                bank.append(item)
            pools[topic, level] = pool
    return bank, pools


class EAP:
    def __init__(self, bank: list[Item], *, use_guessing: bool = False):
        self.use_guessing = use_guessing
        self.grid = tuple(i / 10 for i in range(-80, 81))
        self.prior = tuple(math.exp(-theta**2 / 2) for theta in self.grid)
        self.likelihood = {}
        for item in bank:
            correct = tuple(item.probability(theta, use_guessing=use_guessing) for theta in self.grid)
            self.likelihood[item.index] = (tuple(1 - p for p in correct), correct)

    def update(self, weights: tuple[float, ...], item: Item, answer: int) -> tuple[float, ...]:
        # Общий множитель сокращается в среднем. Для 12 заданий нормировка
        # после каждого ответа не требуется.
        return tuple(w * p for w, p in zip(weights, self.likelihood[item.index][answer]))

    def mean(self, weights: tuple[float, ...]) -> float:
        return math.fsum(theta * w for theta, w in zip(self.grid, weights)) / math.fsum(weights)

    def estimate(self, trace: Trace) -> float:
        weights = self.prior
        for item, answer in trace:
            weights = self.update(weights, item, answer)
        return self.mean(weights)


def run_rules(pools: Pools, ask: Ask, rng: random.Random) -> Trace:
    trace = []
    used = set()
    level = 1
    for topic in TOPIC_ORDER:
        item = rng.choice([item for item in pools[topic, level] if item.index not in used])
        answer = ask(item)
        trace.append((item, answer))
        used.add(item.index)
        level = min(2, level + 1) if answer else max(0, level - 1)
    return trace


def make_mst_panel(pools: Pools, rng: random.Random) -> tuple[list[Item], list[list[Item]]]:
    common = []
    second = [[] for _ in LEVELS]
    for topic in range(len(TOPICS)):
        common_pair = rng.sample(pools[topic, 1], 2)
        common.extend(common_pair)
        for level in range(len(LEVELS)):
            candidates = [item for item in pools[topic, level] if item not in common_pair]
            second[level].extend(rng.sample(candidates, 2))
    return common, second


def run_mst(common: list[Item], second: list[list[Item]], ask: Ask) -> Trace:
    trace = [(item, ask(item)) for item in common]
    correct = sum(answer for _, answer in trace)
    level = int(correct >= 3) + int(correct >= 5)
    trace.extend((item, ask(item)) for item in second[level])
    return trace


def run_cat(by_topic: list[list[Item]], ask: Ask, eap: EAP) -> Trace:
    trace = []
    used = set()
    weights = eap.prior
    for topic in TOPIC_ORDER:
        estimate = eap.mean(weights)
        item = max(
            (item for item in by_topic[topic] if item.index not in used),
            key=lambda item: item.information(estimate, use_guessing=eap.use_guessing),
        )
        answer = ask(item)
        trace.append((item, answer))
        used.add(item.index)
        weights = eap.update(weights, item, answer)
    return trace


@dataclass
class Metrics:
    n: int = 0
    below: int = 0
    false_pass: int = 0
    false_fail: int = 0
    error_sum: float = 0.0
    squared_error_sum: float = 0.0

    def add(self, theta: float, estimate: float) -> None:
        error = estimate - theta
        self.n += 1
        self.error_sum += error
        self.squared_error_sum += error**2
        self.below += theta < THRESHOLD
        self.false_pass += theta < THRESHOLD <= estimate
        self.false_fail += estimate < THRESHOLD <= theta

    def values(self):
        eligible = self.n - self.below
        return (
            self.n,
            math.sqrt(self.squared_error_sum / self.n) if self.n else None,
            self.error_sum / self.n if self.n else None,
            self.false_pass, self.below,
            self.false_pass / self.below if self.below else None,
            self.false_fail, eligible,
            self.false_fail / eligible if eligible else None,
        )


def simulate(
    directory: Path, employees: int, repeats: int, seed: int, near: float,
    response_model: str = "2pl",
) -> None:
    bank, pools = make_bank(random.Random(f"{seed}:bank"), random.Random(f"{seed}:guessing"))
    common, second = make_mst_panel(pools, random.Random(f"{seed}:mst"))
    by_topic = [[item for item in bank if item.topic == topic] for topic in range(len(TOPICS))]
    eap_2pl = EAP(bank)
    eap_3pl = EAP(bank, use_guessing=True)
    use_guessing = response_model == "3pl"
    final_eap = eap_3pl if use_guessing else eap_2pl
    labels = {
        "all": "Все",
        "low": "θ < −1",
        "middle": "−1 ≤ θ < 1",
        "high": "θ ≥ 1",
        "near": f"|θ| ≤ {near:g}",
    }
    metrics = {(group, method): Metrics() for group in labels for method in METHODS}
    with (directory / "bank.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("item_id", "topic", "level", "a", "b", "mst_module", "c"))
        modules = {item.index: "common" for item in common}
        for name, module in zip(("easy", "medium", "hard"), second):
            modules.update((item.index, name) for item in module)
        writer.writerows(
            (item.index + 1, TOPICS[item.topic], LEVELS[item.level], item.a, item.b,
             modules.get(item.index, ""), item.c)
            for item in bank
        )

    with (directory / "tests.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("repeat", "employee", "method", "theta", "theta_hat",
                         "item_ids", "answers"))
        for repeat in range(1, repeats + 1):
            population = random.Random(f"{seed}:population:{repeat}")
            rule_rng = random.Random(f"{seed}:rules:{repeat}")
            thetas = [population.gauss(0, 1) for _ in range(employees)]
            answers = [
                tuple(
                    int(population.random() < item.probability(theta, use_guessing=use_guessing))
                    for item in bank
                )
                for theta in thetas
            ]
            for employee, (theta, row) in enumerate(zip(thetas, answers), start=1):
                def ask(item: Item) -> int:
                    return row[item.index]

                group = "low" if theta < -1 else "high" if theta >= 1 else "middle"
                for method, trace in (
                    ("rules", run_rules(pools, ask, rule_rng)),
                    ("mst", run_mst(common, second, ask)),
                    ("cat", run_cat(by_topic, ask, eap_2pl)),
                    ("cat_3pl", run_cat(by_topic, ask, eap_3pl)),
                ):
                    estimate = final_eap.estimate(trace)
                    metrics["all", method].add(theta, estimate)
                    metrics[group, method].add(theta, estimate)
                    if abs(theta - THRESHOLD) <= near:
                        metrics["near", method].add(theta, estimate)
                    writer.writerow((repeat, employee, method, theta, estimate,
                                     " ".join(str(item.index + 1) for item, _ in trace),
                                     "".join(str(answer) for _, answer in trace)))

    employee_count = f"{employees:,}".replace(",", "\u202f")
    report = [
        "# Сравнение адаптивных тестов",
        "",
        "Банк из 90 заданий. В каждом тесте 12 заданий, по 4 на тему.",
        f"Seed {seed}, повторов {repeats}, сотрудников в каждом повторе {employee_count}.",
        f"Модель генерации ответов и итогового EAP у всех методов — {response_model.upper()}.",
        "",
        "Зачёт при θ̂ ≥ 0. Для каждой ошибки указана доля от числа сотрудников, "
        "для которых она возможна. Прочерк означает пустую выборку для показателя.",
        "",
        "| Группа | Метод | N | RMSE | Средняя ошибка | Ошибочный зачёт | Ошибочный незачёт |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("group", "method", "n", "rmse", "bias",
                         "false_pass", "below_threshold", "false_pass_rate",
                         "false_fail", "at_or_above_threshold", "false_fail_rate"))
        for (group, method), result in metrics.items():
            values = result.values()
            writer.writerow((group, method, *values))
            n, rmse, bias, fp, below, fp_rate, fn, eligible, fn_rate = values
            cells = (
                labels[group], METHODS[method], str(n),
                f"{rmse:.3f}" if rmse is not None else "—",
                f"{bias:+.3f}" if bias is not None else "—",
                f"{fp}/{below} ({fp_rate:.1%})" if fp_rate is not None else "—",
                f"{fn}/{eligible} ({fn_rate:.1%})" if fn_rate is not None else "—",
            )
            report.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    (directory / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--employees", type=int, default=1_000, help="сотрудников в одном повторе")
    parser.add_argument("--repeats", type=int, default=1, help="число независимых выборок")
    parser.add_argument("--seed", type=int, default=42, help="зерно генератора")
    parser.add_argument("--near", type=float, default=0.5, help="полуширина окрестности порога")
    parser.add_argument(
        "--response-model", choices=("2pl", "3pl"), default="2pl",
        help="модель генерации ответов и общего итогового EAP (по умолчанию 2pl)",
    )
    parser.add_argument("--output", type=Path, default=Path("results"), help="каталог для прогонов")
    args = parser.parse_args()
    if args.employees < 1 or args.repeats < 1:
        parser.error("--employees и --repeats должны быть положительными")
    if not math.isfinite(args.near) or args.near <= 0:
        parser.error("--near должен быть конечным положительным числом")

    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".run-", dir=args.output) as staging:
        directory = Path(staging)
        simulate(directory, args.employees, args.repeats, args.seed, args.near, args.response_model)
        result = directory.with_name(directory.name[1:])
        directory.rename(result)
    print(f"Отчёт: {result.resolve() / 'report.md'}")


if __name__ == "__main__":
    main()
