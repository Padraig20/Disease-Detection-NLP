# Disease Detection in Biomedical Free-Text

## 1. Topic

The project revolves around a unique natural language processing challenge where
the primary goal is to recognize specific named entities, especially in
the context of medical conditions and diagnoses.

## 2. Type of Project

A key issue of this project is gathering and annotating data, considering there's
a scarcity of publicly available datasets online for this particular problem. 
Nonetheless, existing medical lexicons, such as Snomed-CT and ICD-10, could 
potentially enhance the model's vocabulary, although I do not know about the
feasibility of this idea.

While there are pre-trained models on Huggingface suitable for similar NER tasks,
they haven't publicly disclosed their training datasets.

## 3. Summary

### a. Idea and Approach

The main idea is free-text processing for extracting diagnoses and diseases from
medical notes. This type of named entity recognition might be useful for e.g.
converting unstructured data to specially constructed standards, suitable for deployment
in Hospital Information Systems.

For this kind of NER project, BERT models are anticipated to be effective. As
such, the plan is to harness BERT base models and adapt them for this specialized task.

### b. Dataset Description

**ESSENTIALS**

The primary dataset originates from the TREC CT topics, publicly accessible
here: http://trec-cds.org/

Each topic has a similar structure, including several diagnoses in free text
format. The topics represent admission notes - notes with the most important
patient details, which a doctor takes as soon as a person is admitted to a 
hospital. This includes personal information and demographics, such as gender and age, but also
and most importantly the current medical conditions, personal medical history and
family medical history. For simplification purposes, the focus lies on
detecting diseases/diagnoses present in the text, covering conditions such as diabetes
mellitus or high blood pressure.

This dataset makes a total of 315 entries (topics). This includes:

- **topics2016.xml** - valuable information in *note*, *description* and *summary*.
50 topics in total. The fields could be processed individually, though, creating
a total of 150 topics.
- **topics2021.xml** - 75 topics in total.
- **topics2022.xml** - 50 topics in total.
- **topics2023.xml** - preprocessed to free text in admission note style via LLM - 40
topics in total.

**ADDITIONALS**

Should the topics dataset not be enough in
case of inference (e.g. error metrics too high), I will include more data from the 
*ClinicalTrials.gov* database. It contains information on clinical trials, including
free text descriptions on said trials. This may be useful to further enhance the
model's performance - given the complexity of annotating this kind of data, I
would consider this only if the model's vocabulary does not suffice.

Since vocabulary in the medical world is complex and diverse, it might be
incredibly useful to enhance the model's vocabulary with already existing
medical thesauri. Some of which (such as ICD-10) are publicly available and
continuously updated by medical professionals. However, I am yet uncertain on
how to incorporate the thesaurus in a BERT's model vocabulary.

**Language**: All data (text) being used in this project will be in English.

### c. Work-Breakdown Structure

- **requirements engineering**
*Goal*: Study and collect requirements for both the project and BERT architecture. 
Possibly find tools on how to efficiently annotate data. Find error metrics suitable
for this task.
*Time*: 5h
*Deadline*: 29th Oct.
- **capturing and annotating data**
*Goal*: Collect all necessary data and make it
ready for training the model.
*Time*: 25h
*Deadline*: 12th Nov.
- **describing data**
*Goal*: Describe data, make plots for data visualization (e.g. wordclouds) for
better understanding of the data we are working with.
*Time*: 5h
*Deadline*: 14th Nov.
- **implementing BERT**
*Goal*: Implement and train a working BERT model.
*Time*: 15h
*Deadline*: 5th Dec.
- **tuning BERT**
*Goal*: Perform Hyperparameter Tuning and detect possible defects.
*Time*: 10h
*Deadline*: 19th Dec.
- **report, presentation and application**
*Goal*: Write a finished report and make a visually appealing presentation. Include a small Angular webapp.
*Time*: 10h
*Deadline*: 16th Jan.
  

## 4. Related Papers and Models

- Named entity recognition and normalization in biomedical literature: a practical case in SARS-CoV-2 literature (https://oa.upm.es/67933/)

This led to the conception of BioBERT, a model refined for disease recognition in biomedical texts. Available at: https://huggingface.co/alvaroalon2/biobert_diseases_ner

- Launching into clinical space with medspaCy: a new clinical text processing toolkit in Python (https://arxiv.org/abs/2106.07799)

The result was a Python package used by medical experts for processing of biomedical texts.
Interestingly enough, this tool can capture many entities and fields that are derived
via rudimentary regular expressions as seen in its source code. Available at: 
https://github.com/medspacy/medspacy

