def map_model_name(model_name):
    name_map = {
        "dgp": "DGP",
        "coxph": "CoxPH",
        "coxnet": "CoxNet",
        "weibullaft": "Weibull",
        "dsm": "DSM",
        "rsf": "RSF",
        "coxboost": "GBSA",
        "deephit": "DeepHit",
        "deepsurv": "DeepSurv",
        "hierarch": "Hierarch.",
        "mtlrcr": "MTLR-CR",
        "mtlr": "MTLR",
        "mensa": "MENSA (Ours)"
    }
    return name_map.get(model_name, model_name)

def map_model_type(model_name):
    model_type_map = {
        "coxph": "SE",
        "coxnet": "SE",
        "weibullaft": "SE",
        "coxboost": "SE",
        "rsf": "SE",
        "mtlr": "SE",
        "deepsurv": "SE",
        "deephit": "CR",
        "dsm": "CR",
        "hierarch": "ME",
        "mensa": "ME"
    }
    key = model_name.lower()
    if key not in model_type_map:
        raise ValueError(f"Unknown model name: {model_name}")
    return model_type_map[key]