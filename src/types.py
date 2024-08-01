from enum import Enum

class EnumBase(Enum):
    @staticmethod
    def get_all_types_names():
        return [d.name for d in list(DataType)]
    
    @staticmethod
    def name2value(name):
        return DataType[name.upper()]
    
    @staticmethod
    def get_associated_type(val):
        if isinstance(val, int):
            return DataType(val)
        elif isinstance(val, str):
            return DataType.name2value(val)
        else:
            raise ValueError(f"Invalid type {type(val)}")

class DataType(EnumBase):
    NUDE = 1
    OBJECT = 2
    VANGOGH = 3


class MethodType(EnumBase):
    EraseDiff = 1
    ESD = 2
    FMN = 3
    Salun = 4
    Scissorhands = 5
    SPM = 6
    UCE = 7
    
class AttackType(EnumBase):
    NTI = 1
    