"""Thin root wrapper for the cross-domain sample-efficiency suite.

Edit the constants below if you want a shorter or longer run.
"""

from teenyreason.multidomain.suite import (
    MultidomainSuiteConfig,
    run_multidomain_suite,
)
from teenyreason.multidomain import (
    ImageProbeBenchmarkConfig,
    LanguageProbeBenchmarkConfig,
)


RL_SEEDS = (0, 1, 2)


if __name__ == "__main__":
    run_multidomain_suite(
        MultidomainSuiteConfig(
            rl_seeds=RL_SEEDS,
            image=ImageProbeBenchmarkConfig(
                label_budgets=(256, 1024, 4096),
                unlabeled_budget=20000,
            ),
            language=LanguageProbeBenchmarkConfig(
                train_char_budgets=(50000, 100000, 200000),
            ),
        )
    )
