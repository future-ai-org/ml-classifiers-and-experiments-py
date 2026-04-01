########################
#
# CREATE THE VOCABULARY
#
########################


from utils import create_vocabulary, open_txt, process_text, sort_and_remove_dups


class VocabularyCreator:

    def __init__(self, filetext):

        # 1. Load a raw text
        self.raw_text = open_txt(filetext)
        print(f"\n✅ TOTAL # OF CHARS FOR RAW TEXT: {len(self.raw_text)}")
        print(f"✅ EXAMPLES: {self.raw_text[:30]}\n")

        # 2. Pre-process all words and signs
        self.preprocessed = process_text(self.raw_text)
        print(f"✅ TOTAL # OF CHARS FOR PREPROCESSED TEXT: {len(self.preprocessed)}")
        print(f"✅ EXAMPLES: {self.preprocessed[:30]}\n")

        # 3. Pre-process by sorting and removing dups
        self.words = sort_and_remove_dups(self.preprocessed)
        print(f"✅ TOTAL # OF SORTED UNIQUE WORDS FOR PREPROCESSED TEXT: {len(self.words)}")
        print(f"✅ EXAMPLES: {self.words[:30]}\n")

        # 4. Create a vocabulary dictionary
        self.vocab = create_vocabulary(self.words)

        print(f"✅ TOTAL # OF ITEMS IN THE VOCABULARY: {len(self.vocab.items())}")
        print("✅ EXAMPLES:")
        for i, item in enumerate(list(self.vocab.items())[-5:]):
            print(          item)

