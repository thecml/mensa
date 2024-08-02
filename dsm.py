from auton_survival import datasets, preprocessing
from auton_survival.models.dsm import DeepSurvivalMachines
from sklearn.model_selection import train_test_split


def train_dsm(train_x, train_t, train_e, text_x, time_grids):
    # train a Deep Survival Machine model and return the predictions (cdf and pdf at the predefined time grids)
    dsm = DeepSurvivalMachines()
    dsm.fit(train_x, train_t, train_e)
    cdf = dsm.predict_risk(text_x, time_grids)
    pdf = dsm.predict_pdf(text_x, time_grids)
    return cdf, pdf


if __name__ == "__main__":
    # Load the Dataset
    features, t, e = datasets.load_dataset("PBC")

    # randomly split the dataset into training and testing sets
    train_x, test_x, train_t, test_t, train_e, test_e = train_test_split(features, t, e, test_size=0.2)
    times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cdf, pdf = train_dsm(train_x, train_t, train_e, test_x, times)
