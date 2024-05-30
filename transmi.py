import argparse
import os
from uroman import Uroman
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import XLMRobertaTokenizer, AutoModelForMaskedLM
import sentencepiece_model_pb2 as sp_model
import glob
import numpy as np
import json
import torch


def get_new_vocab_score(tokenizer_data, origin_to_trans, trans_to_origin, merge_mode):
    
    vocab_score_dict = dict(tokenizer_data['model']['vocab'])
    new_vocab_score_dict = {}
    
    for i in range(len(tokenizer_data['model']['vocab'])):
        transli = origin_to_trans[tokenizer_data['model']['vocab'][i][0]]
        # when the transliteration is not the original vocabulary
        if transli in trans_to_origin:
            if len(trans_to_origin[transli]) == 1:
                score = tokenizer_data['model']['vocab'][i][1]
            else:
                score_list = []
                for source in trans_to_origin[transli]:
                    score_list.append(vocab_score_dict[source])
                if merge_mode == 'max':
                    score = max(score_list)
                elif merge_mode == 'min':
                    score = min(score_list)
                elif merge_mode == "average":
                    score = sum(score_list)/len(score_list)
                else:
                    raise NotImplementedError
        else:
            continue
            score = tokenizer_data['model']['vocab'][i][1]
        new_vocab_score_dict[transli] = score
    new_vocab_score_list = [[x1, x2] for x1, x2 in new_vocab_score_dict.items()]
    new_vocab_score_list = sorted(new_vocab_score_list, key=lambda item: item[1], reverse=True)
    return new_vocab_score_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='xlm-roberta-base')
    parser.add_argument('--merge_mode', default='max')
    parser.add_argument('--save_path', default='./models')
    args = parser.parse_args()
    
    target_path = args.save_path + f"/{args.model_name.split('/')[-1]}-with-transliteration-{args.merge_mode}"
    if os.path.exists(target_path):
        pass
    else:
        os.makedirs(target_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    
    # save the model and the tokenizer on a local 
    tokenizer.save_pretrained(target_path)
    model.save_pretrained(target_path)
    
    #Load pretrained XLM-R SPM
    original_m = sp_model.ModelProto()
    original_m.ParseFromString(open(f"{target_path}/sentencepiece.bpe.model", 'rb').read())
    
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda x: x[1], reverse=False)
    vocab_list = [x[0] for x in vocab]
    
    # get transliterations of all the subwords in the vocab
    roman = Uroman()
    transliterated_vocab = roman.romanize(vocab_list)
    
    # build a dictionary where from the subwords in vocab to their transliteration
    origin_to_trans = dict(zip(vocab_list, transliterated_vocab))
    
    # get the tokens that are not covered by the original vocab
    new_tokens = set(transliterated_vocab).difference(set(vocab_list))
    new_tokens = new_tokens - set(['']) # this fixes bug
    print("Number of newly added tokens: ", len(new_tokens))
    new_tokens = list(new_tokens)
    
    # build a dictionary where the new token is transliterated from
    print("Building transliteration dictionary ...")
    trans_to_origin = {}
    for index, token in enumerate(new_tokens):
        source = []
        for i in range(len(transliterated_vocab)):
            if token == transliterated_vocab[i]:
                source.append(vocab_list[i])
        trans_to_origin[token] = source
        
    # load the tokenizer data through json file 
    with open(f"{target_path}/tokenizer.json", 'r') as file:
        tokenizer_data = json.load(file)
        
    new_vocab_score_list = get_new_vocab_score(tokenizer_data, origin_to_trans, trans_to_origin, args.merge_mode)
    
    
    add_cnt = 0 
    piece_d = {piece.piece: 0 for piece in original_m.pieces}
    for (piece, score) in new_vocab_score_list:
        if piece not in piece_d:
            piece_to_add = sp_model.ModelProto().SentencePiece()
            # Add token
            piece_to_add.piece = piece
            # Add token log-prob
            piece_to_add.score = score
            original_m.pieces.append(piece_to_add)
            add_cnt += 1
    
    # remove all files in the target path
    file_paths = glob.glob(os.path.join(target_path, '*'))
    print(file_paths)
    
    # Loop through and delete each file
    for file_path in file_paths:
        if os.path.isfile(file_path):  # This check ensures you're only deleting files
            os.remove(file_path)

    new_spm_save_dir = f"{target_path}/sentencepiece.bpe.model"
    with open(new_spm_save_dir, 'wb') as f:
        f.write(original_m.SerializeToString())
    
    # store the tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    tokenizer.vocab_file = new_spm_save_dir
    tokenizer.sp_model.load(tokenizer.vocab_file)
    tokenizer.save_pretrained(target_path)
    
    # load necessary information
    tokenizer_old = XLMRobertaTokenizer.from_pretrained(args.model_name)
    tokenizer_new = XLMRobertaTokenizer.from_pretrained(target_path)
    
    token_to_id_dict_old = dict(sorted(tokenizer_old.get_vocab().items(), key=lambda item: item[1]))
    id_to_token_dict_old = {idx: token for token, idx in token_to_id_dict_old.items()}
    
    token_to_id_dict_new = dict(sorted(tokenizer_new.get_vocab().items(), key=lambda item: item[1]))
    id_to_token_dict_new = {idx: token for token, idx in token_to_id_dict_new.items()}
    
    # reinitialize the embeddings
    embeddings = model.get_input_embeddings().weight.detach().numpy()
    new_embeddings = np.zeros((len(token_to_id_dict_new), embeddings.shape[1]), dtype=embeddings.dtype)

    # copy the original embeddings
    for i in range(len(token_to_id_dict_old)):
        idx = token_to_id_dict_new[id_to_token_dict_old[i]]
        new_embeddings[idx] = embeddings[i]

    for i, (token, score) in enumerate(new_vocab_score_list):
        if i % 1000 == 0:
            print(f"{i}, {token}...")
        # get the id of the token
        idx = token_to_id_dict_new[token]
        source_tokens = trans_to_origin[token]
        if len(source_tokens) == 1:
            new_embeddings[idx] = embeddings[token_to_id_dict_old[source_tokens[0]]]
        else:
            if args.merge_mode == 'max':
                # the order is already from the highest to the lowest, so we simply get the first one
                new_embeddings[idx] = embeddings[token_to_id_dict_old[source_tokens[0]]]
            elif args.merge_mode == 'min':
                new_embeddings[idx] = embeddings[token_to_id_dict_old[source_tokens[-1]]]
            elif args.merge_mode == "average":
                emb = np.zeros(embeddings.shape[1])
                for old_indx in [token_to_id_dict_old[source] for source in source_tokens]:
                    emb += embeddings[old_indx]
                new_embeddings[idx] = emb
            else:
                raise NotImplementedError

    model.resize_token_embeddings(len(new_embeddings))
    model.get_input_embeddings().weight.data = torch.from_numpy(new_embeddings)
    model.save_pretrained(target_path)
    
    print(model)

