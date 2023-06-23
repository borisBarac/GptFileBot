from enum import Enum
from langchain.llms import OpenAI

class ObjectiviTyLevel(Enum):
    NORMAL = 0
    FILE_BASED = 1

    def get_temperature(self):
        if self.value == 0:
            return 0.1
        elif self.value == 1:
            return 0.9
        else:
            return 0.1

def make_llm(objectivity: ObjectiviTyLevel) -> OpenAI:
    llm = OpenAI(temperature= objectivity.get_temperature(),
                #  model='text-babbage-001',
                 max_tokens=50,
                 verbose=True)


    return llm