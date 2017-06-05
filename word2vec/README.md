

# Related Works
https://github.com/candlewill/Chinsese_word_vectors	a github repository trying to built a chinese word2vec

https://github.com/Kyubyong/wordvectors			a github repository having pretrained 30+ languages word2vecs, may be performed by a South Korea Professional or other contry's.

http://www.52nlp.cn/%E4%B8%AD%E8%8B%B1%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%E8%AF%AD%E6%96%99%E4%B8%8A%E7%9A%84word2vec%E5%AE%9E%E9%AA%8C					at 52NLP, a blog that describes how to perform a chinese word2vec training task, related github: https://github.com/panyang/Wikipedia_Word2vec

http://blog.csdn.net/lixintong1992/article/details/50387007	A more simple guide to perform the task mentioned above, including transform traditional Chinese to simplified Chinese used by the mainland of China.

http://www.cnblogs.com/hebin/p/3507609.html	a word2vec trainer using 2.1 G different soruce texts

# Suggest
When using model.save('...'), do not contain ':' in your file.Or it will cause a protocal error,very weired.
