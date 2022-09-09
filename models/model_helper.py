import importlib


def build_network(config):
    mtype = config["builder"]
    module_name, fn_name = mtype.rsplit(".", 1)
    module = importlib.import_module(module_name)
    model_builder = getattr(module, fn_name)
    kwargs = config.get("kwargs", {})
    model = model_builder(**kwargs)
    return model
