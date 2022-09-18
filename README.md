# Machine Learning Applied to Health
My laboratories from first year subject Msc. ICT for Smart Societies, subject: ICT for health 2021/22. Politecnico di Torino, Italy.
@zamora

# Report 1: Regression on Parkinson’s disease data
Patients affected by Parkinson’s disease cannot perfectly control their muscles. In particular
they show tremor, they walk with difficulties and, in general, they have problems in starting
a movement. Many of them cannot speak correctly, since they cannot control the vocal
chords and the vocal tract.
Levodopa is prescribed to patients, but the amount of treatment should be increased
as the illness progresses and it should be provided at the right time during the day, to
prevent the freezing phenomenon. It would be beneficial to measure total UPDRS (Unified
Parkinson’s Disease Rating Scale) many times during the day in order to adapt the treatment
to the specific patient. This means that an automatic way to measure total UPDRS should
be developed, using simple techniques easily managed by the patient or his/her caregiver.
One possibility is to use patient voice recordings (that can be easily obtained several
times during the day through a smartphone) to generate vocal features that can be then
used to regress total UPDRS.
In the following, linear regression is used, with three different algorithms, and it is applied
on the public dataset.
# Report 2: Gaussian Process Regression on Parkinson’s
For the last problem, Gaussian Process Regression (GPR) was used on the public dataset to estimate total
UPDRS, and the results were compared to those obtained with linear regression, showing
the superiority of GPR.
# Report 3: COVID-19: analysis of two serological tests
Spreading of COVID-19 (COrona VIrus Disease 19) infection can be reduced with early
detection of infected people, so that they can start quarantine as soon as possible. The
nasopharyngeal swab test is highly reliable but it requires time and is expensive, serological
tests are faster and cheaper, but less reliable. Serological tests find the presence of IgG
(Immunoglobulin G) and a high level of this antibody in blood means that the person is or
has been affected by COVID-19.
This work reports the results of the analysis of two serological tests, discussing the setting
of the thresholds to declare a positive result.
# Report 4: Clustering techniques for COVID-19 CT scan analysis
With the increasing prevalence of coronavirus disease-19 (COVID-19) infection worldwide,
early detection has become crucial to ensure rapid prevention and timely treatment. However, due to the unknown gene sequence of the supposed coronavirus, the reference standard
test has not been established for diagnosis. Several studies have suggested pneumonia as the
underlying mechanism of lung injury in patients with COVID-19 Accordingly, it is believed
that the pulmonary lesions caused by COVID-19 infection are similar to those of pneumonia. More than 75% of suspected patients showed bilateral pneumonia. In this context, the
promising findings of several studies have highlighted the growing role of chest computed
tomography (CT) scan for identifying suspected or confirmed cases of COVID-19 infection.
The common typical chest CT scan findings are summarized as: Peripheral distribution,
Bilateral lung involvement, Multifocal involvement, Ground glass opacification-GGO (instead of appearing uniformly dark), Crazy paving appearance (appearance of ground-glass
opacity with superimposed interlobular septal thickening and intralobular septal thickening),
Interlobular septal thickening(numerous clearly visible septal lines usually indicates the presence of some interstitial abnormality), Bronchiolectasis (dilatation of the usually terminal
bronchioles (as from chronic bronchial infection)). In other words, lung alveoli are partially
filled with exudate or they are partially collapsed and the tissue around alveoli is thickened.
Not all the patients affected by COVID-19 show interstitial pneumonia, but its presence
is a fast way to diagnose COVID-19. Nasopharyngeal swab analysis requires some hours in
the lab plus the time to deliver the swab to the lab; on the contrary, any hospital has CT
scanners and the radiologist can immediately detect the presence of ground glass opacities.
However, it would be useful to design an algorithm to help radiologists in this task. In the
next sections a method is described that identifies these opacities for the subsequent analysis
by the radiologist. The software was developed in Python, using the Scikit-learn library.
# Report 5: Chronic Kidney Disease classification
Chronic kidney disease (CKD) derives from a gradual loss of kidney filtering capability over
time, typically caused by high blood pressure and diabetes. Prevalence of the illness is
around 10% in adult population, and its early detection avoids the dramatic consequence of
complete kidney failure and necessity of kidney transplant.
Whilst a cure does not exist for CKD, treatments of kidney disease are available to reduce
the symptoms, but they are expensive and impair the normal life of the affected subject (long
dialysis sessions).
Kidney functionality can be assessed through the Glomerular Filtration Rate (GFR),
calculated from the 24-hour collected urine or from the blood creatinine test.
A public dataset to explore correlations between CKD and subject parameters. In particular, the dataset includes 24 features, among which 11 are
numerical and 13 are categorical. Each of the 400 points of the dataset belongs either to
class ckd (chronic kidney disease is present) or notckd. Unfortunately, some features are
missing for some subjects and must be replaced; on the contrary, there are no
cases of missing class.
Object of the work is to use the dataset to build decision trees to classify new subjects as
either healthy or affected by chronic kidney disease and measure the performance. Decision
trees are all built using Python Scikit Learn class DecisionTreeClassifier [2] using entropy
criterion; missing values are replaced using regression trees available in the same Python
library.
