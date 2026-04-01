####################################
#
#   Entry point of this project
#
###################################


if __name__ == "__main__":

    from activation_functions import print_example_dnn, print_gradients_example
    from create_vocab import VocabularyCreator
    from gpt_dataloader import create_dataloader, print_batch_example
    from gpt_model import (
        GPTModel,
        GPTModel2,
        generate_text_example,
        prepare_input_data,
        print_logits,
        print_tranformer_block,
    )
    from multihead_attention import (
        MultiHeadAttentionWrapper,
        simple_mha_example,
    )
    from pretraining import run_pretraining_example
    from self_attention import (
        CausalAttention,
        SelfAttention,
        SelfAttentionManual,
        get_context_vector,
        print_self_attention_nn,
    )
    from simple_tokenizer import SimpleTokenizer
    from token_embedding import print_embedding_example


    FILETEXT = "data/the-verdict.txt"


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING THE VOCAB \n')
    vocab_creator = VocabularyCreator(FILETEXT)
    raw_text = vocab_creator.raw_text
    vocab = vocab_creator.vocab


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING THE TOKENIZER \n')
    tokenizer = SimpleTokenizer(vocab)
    tokenizer.example()


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING THE DATALOADER \n')
    dataloader = create_dataloader(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    print_batch_example(dataloader)


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING THE TOKEN EMBEDDING \n')
    print_embedding_example(dataloader, vocab_size=50257, output_dim=256)


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING THE ATTENTION EXAMPLE \n')
    self_attention = SelfAttentionManual()
    self_attention.simple_example()


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING THE ATTENTION EXAMPLE WITH TRAINABLE WEIGHTS\n')
    self_attention.trainable_weights_example()


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING THE ATTENTION EXAMPLE WITH PYTORCH\n')
    self_attention = SelfAttention()
    print_self_attention_nn(self_attention)


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING CAUSAL ATTENTION\n')
    causal_attention = CausalAttention(dropout=0.0)
    get_context_vector(causal_attention)


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING MULTIHEAD ATTENTION\n')
    mha = MultiHeadAttentionWrapper(dropout=0.0, num_heads=2)
    get_context_vector(mha)
    simple_mha_example()


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> EXAMPLE DNN\n')
    #plot_relu_vs_gelu()
    print_example_dnn()
    print_gradients_example()


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> CREATING GPT MODEL\n')
    batch = prepare_input_data()
    print_tranformer_block()

    model1 = GPTModel()
    print_logits(model1, batch)
    generate_text_example(model1)

    model2 = GPTModel2()
    print_logits(model2, batch)
    generate_text_example(model2)


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> PRETRAINING ON UNLABELED DATA\n')
    run_pretraining_example(model1)
    run_pretraining_example(model2)


    print ('\n\n\n--------------------------------------------------------------')
    print ('--------------------------------------------------------------')
    print('----> PRETRAINING ON UNLABELED DATA\n')
    run_pretraining_example(model1)
    run_pretraining_example(model2)
