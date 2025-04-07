
MODEL_REGISTRY = {}


def register_model(model_class):
    MODEL_REGISTRY[model_class.__name__] = model_class
    return model_class
