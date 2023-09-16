import os
import re
import json
import numpy as np
from collections import defaultdict


def make_vocab_questions(input_dir):
    """Make dictionary for questions and save them into text file."""
    vocab_set = set()
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
    question_length = []
    datasets = os.listdir(input_dir)
    for dataset in datasets:
        with open(input_dir + '/' + dataset) as f:
            questions = json.load(f)['questions']
        set_question_length = [None] * len(questions)
        for iquestion, question in enumerate(questions):
            words = SENTENCE_SPLIT_REGEX.split(question['question'].lower())
            words = [w.strip() for w in words if len(w.strip()) > 0]
            vocab_set.update(words)
            set_question_length[iquestion] = len(words)
        question_length += set_question_length

    vocab_list = list(vocab_set)
    vocab_list.sort()
    vocab_list.insert(0, '<pad>')
    vocab_list.insert(1, '<unk>')

    with open(r'C:\Users\hieunc15\Documents\VisualQA\data\processed\vocab_questions.txt', 'w') as f:
        f.writelines([w + '\n' for w in vocab_list])

    print('Make vocabulary for questions')
    print('The number of total words of questions: %d' % len(vocab_set))
    print('Maximum length of question: %d' % np.max(question_length))


def make_vocab_answers(input_dir, n_answers):
    """Make dictionary for top n answers and save them into text file."""
    answers = defaultdict(lambda: 0)
    datasets = os.listdir(input_dir)
    for dataset in datasets:
        with open(input_dir + '/' + dataset) as f:
            annotations = json.load(f)['annotations']
        for annotation in annotations:
            for answer in annotation['answers']:
                word = answer['answer']
                if re.search(r"[^\w\s]", word):
                    continue
                answers[word] += 1

    answers = sorted(answers, key=answers.get, reverse=True)
    assert ('<unk>' not in answers)
    top_answers = ['<unk>'] + answers[:n_answers - 1]  # '-1' is due to '<unk>'

    with open(r'C:\Users\hieunc15\Documents\VisualQA\data\processed\vocab_answers.txt', 'w') as f:
        f.writelines([w + '\n' for w in top_answers])

    print('Make vocabulary for answers')
    print('The number of total words of answers: %d' % len(answers))
    print('Keep top %d answers into vocab' % n_answers)
