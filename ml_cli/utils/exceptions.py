class MLCliError(Exception):
    """Base exception for ML CLI"""
    pass

class ConfigurationError(MLCliError):
    """Configuration related errors"""
    pass

class DataError(MLCliError):
    """Data related errors"""
    pass

class ModelError(MLCliError):
    """Model related errors"""
    pass
