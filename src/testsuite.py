import unittest
import numpy as np
import torch
from utils.dataloader import Dataloader, tokenize_and_preserve_labels
from utils.metric_tracking import MetricsTracking 
from transformers import BertTokenizer
import re

class DataloaderTest(unittest.TestCase):

    def test_tokenize_sentence(self):
        label_to_ids = {
           'B-MEDCOND': 0,
           'I-MEDCOND': 1,
           'O': 2
        }
        ids_to_label = {
               0:'B-MEDCOND',
               1:'I-MEDCOND',
               2:'O'
        }
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_tokens = 128

        sentence = "Patient presents with glaucoma, characterized by a definitive diagnosis of pigmentary glaucoma. Intraocular pressure measures at 15 mmHg, while the visual field remains normal. Visual acuity is recorded as 20/50. The patient has not undergone prior cataract surgery, but has had LASIK surgery. Additionally, comorbid ocular diseases include macular degeneration."

        tokens = "O O O B-MEDCOND O O O O O O O B-MEDCOND I-MEDCOND O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-MEDCOND I-MEDCOND I-MEDCOND O B-MEDCOND I-MEDCOND O"

        sentence = re.findall(r"\w+|\w+(?='s)|'s|['\".,!?;]", sentence.strip(), re.UNICODE)
        tokens = tokens.split(" ")

        t_sen, t_labl = tokenize_and_preserve_labels(sentence, tokens, tokenizer, label_to_ids, ids_to_label, max_tokens)

        self.assertEqual(len(t_sen), len(t_labl))
        self.assertEqual(t_labl.count("B-MEDCOND"), tokens.count("B-MEDCOND"))

    def test_load_dataset(self):
        label_to_ids = {
           'B-MEDCOND': 0,
           'I-MEDCOND': 1,
           'O': 2
        }
        ids_to_label = {
               0:'B-MEDCOND',
               1:'I-MEDCOND',
               2:'O'
        }

        dataloader = Dataloader(label_to_ids, ids_to_label)

        dataset = dataloader.load_dataset(full = True)

        self.assertEqual(len(dataset), 255)

        sample = dataset.__getitem__(0)

        self.assertEqual(len(sample), 4) #input_ids, attention_mask, token_type_ids, entity



class MetricsTrackingTest(unittest.TestCase):

    def test_avg_metrics(self):
        predictions =  np.array([-100, 0, 0, 0, 1, 1, 1, 2, 2, 2])
        ground_truth = np.array([-100, 0, 1, 0, 1, 2, 1, 2, 0, 2]) #arbitrary, should return 67% for each metric
        
        predictions = torch.from_numpy(predictions)
        ground_truth = torch.from_numpy(ground_truth)

        tracker = MetricsTracking()
        tracker.update(predictions, ground_truth, 0.1)

        metrics = tracker.return_avg_metrics(1) #tracker only updated once

        self.assertEqual(metrics['acc'], 0.667)
        self.assertEqual(metrics['f1'], 0.667)
        self.assertEqual(metrics['precision'], 0.667)
        self.assertEqual(metrics['recall'], 0.667)
        self.assertEqual(metrics['loss'], 0.1)

if __name__ == '__main__':
    unittest.main()
