# NHS England Splink Linkage Implementation Model Card

_For more details of model cards and how we use them in the data science team in NHS England, see the [NHS England Template Model Card ](https://github.com/nhsengland/model-card)._

> [!NOTE]
> Please note that this model card refers to the implementation of Splink used for _linking_ records. A separate model card will describe the implementation of Splink used to de-duplicate records. These are referred to as 'link_only' and 'dedupe_only' respectively.

## Specification

|  |  |
| ---- | ---- |
| **Description:** | This repository uses [Splink][1] to train, run, and evaluate linkage models that can be used to link datasets to the [Personal Demographics Service][2] (PDS) database.
| **Model Type:** | Probabilistic linkage (Fellegi-Sunter) model|
| **Developed By:** | NHSE Data Linkage Hub & Data Science Team |
| **Launch Date:** | TBC |
| **Version** | 0.3 | # (this relates to MVP as on confluence docs, but this is not currently linked to github releases)

## Intended Use

**Development Background:** Splink, an open-source probabilistic linkage package developed by the UK's Ministry of Justice, was identified as a tool that could be an alternative to [Master Person Service][3], the deterministic linkage algorithm currently used for linking many NHS England datasets to PDS. This repository contains code which adapts Splink to operate on NHS England's digital infrastructure, includes preprocessing specifically relevant to NHS England datasets, and Splink model parameters suitable for NHS England datasets.

**Scope:** Linking NHS personal data to PDS.

**Intended Users:** Linkers of data in or working with the NHS in England.

**Use cases out of scope:** Linking of other types of personal data.

## Data

**Data Overview:** The [select_training_data](/notebooks/select_training_data_linking.py) contains code used to select training and evaluation data.

**Sensitive Data:** This model is designed to be trained on and used with sensitive data. People with access to this data are kept to a minimum and all code and analysis undertaken on a secure NHS data platform.

**Pre-processing and cleaning:**

Code for pre-processing can be found in [utils](/utils/preprocessing_utils.py).

In summary:

* Confidential records- Records with confidentiality codes applied are removed from data to ensure we do not include those whose data should not be linked for legal and security reasons.
* Names- Given and family names are processed to account for obvious error, e.g. removing special characters apart from dashes and periods. .
* Postcodes- Confusion between zeros and the letter 'o' is handled according to the position of the character. Invalid postcodes are replaced with an invalid code.

**Data Split:** TBC

## Methodology and Training

**Model Type:** Probabilistic (Fellegi-Sunter) linkage model. For more details see the Splink documentation[^1].

**Justification:** The current linkage method used in MPS is relatively opaque. Probabilistic models provide a ranking of matching decisions driven by associations in the training data, that can be explained through to the linked, or unlinked result.

**Algorithm Details:** $m$ & $u$ parameters, distance metrics and blocking rules have been optimised (see more details in [the training methods section](#training-methods)), and then used as parameters of a Splink model to predict links between datasets and PDS.

**Alternative Methods Considered:** TBC (are other libraries available?)

### Training Methods:

**Training Process:** The [training notebook](/notebooks/training.py) contains the code to train the model.

**Hyperparameter/Fine Tuning:** The following parameters have been optimised:
* $m$ & $u$ parameters
* comparisons and comparison levels (see the [Splink docs for more info about comparisons][4])
* blocking rules

## Evaluation and Performance

### Model Evaluation

**Evaluation Process:** The [evaluation notebook](/notebooks/evaluation.py) contains the code to evaluate the model.

**Evaluation Focus:** Initially, the outputs of this model were compared to the results of the MPS linkage service to see what the differences in performance were. Many of these differences were assessed manually.

**Performance breakdown:** TBC

**Metrics:** Performance metrics such as F1 Score, Precision or Recall may not be appropriate for Data Linkage where a ground truth is not necessarily known. Work is ongoing to establish what linkage quality metrics to use in this area.

**Performance in Deployment**: TBC

### Ethical Considerations

**Bias and fairness analysis:** Some analysis has been carried out internally to understand the differences between linkable and unlinkable (no records found that meet a probability threshold). Some non-insignificant differences were found, which should be communicated to end users of linked data depending on the specifics of their input data. Work is underway to establish what metadata is required here.

**Implications for human safety:** Records that cannot be linked will then likely not be included in subsequent work and analysis, which carries the risk of certain groups that are less likely to be linkable are excluded and therefore decisions (such as policy decisions) are not informed by their experiences.

### Caveats

**Caveats and Limitations:** _Any areas we know this underperforms._

[1]: https://moj-analytical-services.github.io/splink/index.html
[2]: https://digital.nhs.uk/services/personal-demographics-service
[3]: https://digital.nhs.uk/services/personal-demographics-service/master-person-service
[4]: https://moj-analytical-services.github.io/splink/topic_guides/comparisons/customising_comparisons.html
