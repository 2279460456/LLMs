text = "Natural Language Processing (NLP) stands as a pivotal and exceptionally dynamic domain within the broader fields of computer science, artificial intelligence, and computational linguistics. It is primarily concerned with enabling computers to comprehend, interpret, and generate human language in a way that is both meaningful and useful. The overarching ambition of NLP is to bridge the communication gap between humans and machines, allowing for seamless and intuitive interactions. Modern NLP heavily relies on sophisticated machine learning models, including deep learning architectures like recurrent neural networks (RNNs) and transformers. These models are trained on vast corpora of text and speech data, learning intricate patterns, grammatical structures, and semantic relationships. The initial and crucial phase in most NLP systems is tokenization, where input text is segmented into smaller, manageable units such as words, subwords, or characters. Subsequent to tokenization, a cascade of analytical techniques can be employed. These may include morphological analysis (stemming and lemmatization to reduce words to their root forms), syntactic analysis (part-of-speech tagging and parsing to understand grammatical structure), and semantic analysis (named entity recognition, word sense disambiguation, and sentiment analysis to extract meaning). The advent of pre-trained large language models (LLMs) like BERT, GPT, and their successors has revolutionized the NLP landscape. These models, often containing billions of parameters, exhibit remarkable proficiency in a wide array of tasks, including high-quality machine translation across numerous languages, coherent and contextually relevant text summarization, accurate question answering over large documents, realistic dialogue generation, and even the creation of original written content. Despite these monumental advancements, NLP continues to face significant hurdles. Effectively resolving ambiguity inherent in human language, imbuing systems with genuine common sense reasoning capabilities, understanding and responding appropriately to figurative language (such as sarcasm, irony, and metaphor), and ensuring fairness and mitigating biases in NLP models are active areas of ongoing research and development. Furthermore, adapting NLP techniques to low-resource languages and specialized domains remains a persistent challenge that the community is actively working to address. Let's add some fun characters: 😃👍🎉 αβγ multilingual text like 你好 and 안녕하세요. Also some symbols: ©™®"
tokens = text.encode('utf-8')
tokens = list(map(int,tokens))

# print('-----------')
# print(text)
# print("length:",len(text))
# print('-----------')
# print(tokens)
# print("length:",len(tokens))

#------------------------------------------
# 获取所有pair出现的次数
def get_stats(ids):
  counts = {}
  for pair in zip(ids,ids[1:]):
    counts[pair] = counts.get(pair,0)+1
  return counts

# stats = get_stats(tokens)
# print(sorted(((v,k) for k,v in stats.items()),reverse=True))

#------------------------------------------
# 合并出现频率最高的一项
# top_pair = max(stats,key=stats.get)
# print(top_pair)

#------------------------------------------
# 对ids进行合并，将pair对合并为idx
def merge(ids,pair,idx):
  newIds = []
  i = 0
  while i<len(ids):
    if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
      newIds.append(idx)
      i+=2
    else:
      newIds.append(ids[i])
      i+=1
  return newIds
# tokens2 = merge(tokens,top_pair,256)
# print(tokens2)
# print("tokens2 length:",len(tokens2))
# print("tokens1 length:",len(tokens))

#------------------------------------------
# 进行多次合并
vocab_size = 276
num_merges = vocab_size-256

merges={}
for i in range(num_merges):
  stats = get_stats(tokens)
  top_pair = max(stats,key=stats.get)
  idx = 256+i
  # print(f"merging {top_pair} into a new token {idx}")
  tokens = merge(tokens,top_pair,idx)
  merges[top_pair] = idx

# print(f"完成{num_merges}次合并的结果：{tokens}。长度为：{len(tokens)}")
# print(f"合并过程：{merges}")

#------------------------------------------
#decoding
vocab = {idx:bytes([idx]) for idx in range(256)}
for (p0,p1),idx in merges.items():
  vocab[idx] = vocab[p0]+vocab[p1]

def mydecode(tokens):
  tokens = b"".join([vocab[token] for token in tokens])
  return tokens.decode(encoding='utf-8',errors='replace')

# print(mydecode(tokens))

#------------------------------------------
#encoding
def myencode(text):
  tokens = list(text.encode('utf-8'))
  while len(tokens)>=2:
    stats = get_stats(tokens)
    pair = min(stats,key=lambda p:merges.get(p,float('inf')))
    if pair not in merges:
      break
    idx = merges[pair]
    tokens = merge(tokens,pair,idx)
  return tokens

print(myencode(text))





