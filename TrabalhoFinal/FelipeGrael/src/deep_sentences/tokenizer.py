from itertools import chain

from datasets import Dataset
from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing



def train_tokenizer(train: Dataset) -> Tokenizer:

    # Create a simple whitespace tokenizer
    tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(behavior="removed"),
    ])

    trainer = WordLevelTrainer(special_tokens=["<PAD>", "<EOS>", "<UNK>"])

    all_sentences = chain(train['sentence1'], train['sentence2'])

    tokenizer.train_from_iterator(all_sentences, trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="$A <EOS>",
        special_tokens=[("<EOS>", tokenizer.token_to_id("<EOS>"))]
    )

    return tokenizer


# Tokenize
def tokenize_dataset(dataset: Dataset, tokenizer: Tokenizer):

    def tokenize_row(row):
        return {
            'sentence1_ids': tokenizer.encode(row['sentence1']).ids,
            'sentence2_ids': tokenizer.encode(row['sentence2']).ids
        }

    return dataset.map(tokenize_row, remove_columns=['sentence1', 'sentence2'])
