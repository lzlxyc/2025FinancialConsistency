import abc

class BaseAgent(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self):
        pass