def map_model_name(model_name):
    if model_name == "dgp":
        model_name = "DGP"
    elif model_name == "dsm":
        model_name = "DSM"
    elif model_name == "deephit":
        model_name = "DeepHit"
    elif model_name == "deepsurv":
        model_name = "DeepSurv"
    elif model_name == "hierarch":
        model_name = "Hierarch."
    elif model_name == "mtlrcr":
        model_name = "MTLR-CR"
    elif model_name == "mtlr":
        model_name = "MTLR"
    elif model_name == "mensa":
        model_name = "MENSA"
    else:
        pass
    return model_name