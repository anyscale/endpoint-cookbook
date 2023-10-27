from .hotpotqa import HotpotQATask
from .strategyqa import StrategyQATask
from .mmlu import MMLUTask
from .bamboogle import BamboogleTask
from .triviaqa import TriviaQATask


def get_task(name, split):
    if name == "hotpotqa":
        return HotpotQATask(split)
    elif name == "strategyqa":
        return StrategyQATask(split)
    elif name == "mmlu":
        return MMLUTask(split)
    elif name == "bamboogle":
        return BamboogleTask(split)
    elif name == "triviaqa":
        return TriviaQATask(split)
    else:
        raise ValueError("Unknown task name: {}".format(name))
    
