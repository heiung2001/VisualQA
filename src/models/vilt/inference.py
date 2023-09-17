import torch
from transformers import ViltProcessor, ViltForQuestionAnswering


class ViLTInference:
    def __init__(self):
        weight = "dandelin/vilt-b32-finetuned-vqa"
        print("Loading: {}".format(weight))

        self.processor = ViltProcessor.from_pretrained(weight)
        self.model = ViltForQuestionAnswering.from_pretrained(weight)

    def __call__(self, image, text):
        encoding = self.processor(image, text, return_tensors='pt')

        outputs = self.model(**encoding)
        logits = outputs.logits
        predicted_classes = torch.sigmoid(logits)

        answer_dict = dict()
        probs, classes = torch.topk(predicted_classes, 5)
        for prob, classes_idx in zip(probs.squeeze().tolist(), classes.squeeze().tolist()):
            answer_dict[self.model.config.id2label[classes_idx]] = prob

        return answer_dict
