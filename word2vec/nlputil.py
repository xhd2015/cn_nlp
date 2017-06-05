#encoding=utf-8
'''
	python version: 2.7
	providing functions that very useful in pre-processing and processing Chinese documents
'''
import sys
import jieba
import re
import zhconv
import bz2
import os.path
import gensim.corpora
import gensim
import time

import itertools
import functools

import cPickle

try:
	import pynlpir
	pynlpir.open()
except:
	pynlpir=None

def convertToNonStopsent(sent,stopwords=None):
	'''
		:param	sent:
			a list of word,not a joint string
	'''
	return list(filter(functools.partial(nonStopwordsFilter,stopwords=stopwords),sent))
def catFiles(target,*files):
	def _bopen(file):
		return open(file,'rb')
	iterfiles = itertools.imap(_bopen,files)
	outf = open(target,'wb')
	for f in iterfiles:
		outf.write(f.read())
		f.close()
	outf.close()

	
def load_pickle(filepath):
        f=open(filepath,'rb')
        data=cPickle.load(f)
        f.close()
        return data
def dump_pickle(data,filepath):
        with open(filepath,'wb') as f:
                cPickle.dump(data,f)

def clearPS():
	sys.ps2=''
def restorePS():
	sys.ps1='>>> '
	sys.ps2="... "
def loadWord2vec(fname):
	return gensim.models.Word2Vec.load(fname)

def getCurrentTimestamp():
	return time.strftime('%Y_%m_%d_%H_%M_%S')
######################################################
def regexChecker(regex,text):
	'''
		:param	regex
			a regex or a string
	'''
	if type(regex)==type(''):
		regex=re.compile(regex)
	return regex.match(text) is not None
def yesNoChecker(text):
	return text.lower() in ['yes','no','y','n']
def intChecker(text):
	return regexChecker('\\d+',text)
def fileChecker(text):
	return os.path.isfile(text)
def dirChecker(text):
	return os.path.isdir(text)
def trueChecker(text):
	return True
def askUntil(prompt,checker=trueChecker,duration=2*60.0,re_prompt='Input error, try again',default=''):
	'''
		ask for user's input until:
			duration ended
			input checked
		return 
			default	if ENTER or timedout
			None	if CTRL-C
	'''
	print('You\'ll have {} minutes to finish the following input.'.format(duration/60.0))
	startx=time.time()
	while time.time() - startx < duration:
		try:
			intext=raw_input('{}(type ENTER to get default value "{}")'.format(prompt,default)).strip()
			if len(intext)==0:
				return default
			if not checker(intext):
				print(re_prompt)
			else:
				return intext
		except KeyboardInterrupt,e:
			return None #interrupted
		except:
			print(re_prompt)
	return default
############################################################


############################################################
###	utilities for evaluating model		###########
def unique_reduce(data,unique_key=lambda x:x):
	'''
		unique the dataset by selecting unique key.
		:return
			items of list where each item in list has the same unique_key
	'''
	last_key=None
	new_data = []
	j_new = -1
	for i,iter_data in enumerate(data):
		cur_key = unique_key(iter_data)
		if last_key == cur_key:
			new_data[j_new].append(iter_data)
		else: #last_key is not None or is None
			new_data.append([iter_data])
			j_new += 1
			last_key = cur_key
	return new_data
def eval_model_data( data, unique_key = lambda x:x[0], sort_key = lambda x:x[0]) :
	'''
		prepare data for eval
		:param data
			data to be zipped,i.e zip(*data) will be used
	'''
	reduced_data = unique_reduce( zip(*data), unique_key=unique_key )
	def sort_return(data):
		data.sort(key= sort_key)
		return data
	reduced_data = map(sort_return,reduced_data)
	return reduced_data
	
def eval_MRR(ids,labels,sims):
	'''
		:param	ids:
			id of each corrsponding question
		:param	labels:
			label if each corresponding answer
		:param	sims
			similarity of each corresponding QA pair
		algorithm:
			init mrr=0, num=0
			for each unique question id,get all its correspinding answer set,sort sims with index recorded.
			if answer set contains no answer,then it is mrr+=0
			if answer set contains exactly 1 answer, then mrr+=1/rank
			if answer set contains more that 1 answer,then mrr+=max(1/rank) i.e. the first meet index
			for each id, num+=1
		NOTE:
			if there are multiple answers marked as 1 for one question, then MRR will be abnormally high compared with accuracy.
			A better idea is to return the best MRR & the worst MRR.
			Or if you want to get a good MRR, you should gurantee that each question has at least one answer with your best
	'''
	reduced_data = eval_model_data( (ids,labels,sims), unique_key=lambda x:x[0], sort_key=lambda x:-x[2])
	mrr=0.0
	sum_data = float(len(reduced_data))
	if sum_data==0.0:
		mrr=0.0
	else:
		for batch in reduced_data:
			for index,(_,label,_) in enumerate(batch,1):
				if label==1:
					#print 'index is:',index
					mrr += 1.0/index
					break
		mrr /= sum_data
	return mrr

def eval_accuracy(ids,labels,sims):
	'''
		evaluate the number of top 1 of sims and that of labels shared.
		OR explained as:
			the ratio of hit of the correct answer
	'''
	reduced_data = eval_model_data( (ids,labels,sims), unique_key=lambda x:x[0], sort_key=lambda x:-x[2])
	sum_data = float(len(reduced_data))
	sum_right = 0
	acc = 0.0
	if sum_data == 0.0:
		acc = 0.0
	else:
		for batch in reduced_data:
			if len(batch)>0 and batch[0][1]==1:
				sum_right += 1
		acc = sum_right/sum_data	
	return acc
#######################################################################################

def pynlpirCutter(sent):
	if pynlpir:
		return pynlpir.segment(sent,pos_tagging=False)
	else:
		raise Exception('Pynlpir cannot be used')

def toSimplified(word):
	'''
		transform a word of unicode or str to simplified unicode chinese word
	'''
	if type(word)!=type(u''):
		word=word.decode('u8')
	return zhconv.convert(word,'zh-cn')
def getChangedFname(fname,addon):
	'''
		fname='a.txt',addon='.fake'
		return 'a.fake.txt'
	'''
	x=os.path.splitext(fname)
	return x[0]+addon+x[1]
def saveZhWikiLineSentence(fin,fout=None,usezip=None,stopfile=None):
	'''
		possible usezip values: bz2,None
		return:
			the true out file name is returned.
	'''
	if fout is None:
		fout=getChangedFname(fin,'.LineSentence')
	keys=openerMap.keys()
	if usezip not in openerMap:
		raise Exception('zip format not supported or incorrect,NOTE supported'+str(keys))
	if usezip is not None and not fout.endswith('.'+usezip):
		fout+='.'+usezip
	else:
		for key in keys:
			if key is not None:
				if fout.endswith('.'+key):
					fout+='.notzipped'
					break
	stopwords=None
	if stopfile is not None:
		stopwords=loadStopWords(stopfile)
	wiki=gensim.corpora.WikiCorpus(fin,lemmatize=False,dictionary={})
	with openerMap[usezip](fout,'wt') as out:
		for line in zhTextToLineSentence(wiki.get_texts(),stopwords=stopwords):
			##DEBUG
			#print(line)
			out.write(line.encode('u8'))
	return fout
	
def zhTextToLineSentence(text_iter,cutter=jieba.cut,keep_punct=True,tosimplified=toSimplified,stopwords=None,keep_others=False):
	'''
		transfer a zhwiki to LineSentence, a wrapper for zhWikiProcess, see it for more details.
		example:
			g=utils.zhWikiToLineSentence(wiki,stopwords=None,keep_others=False)
			for line in g:
				f.write(line)
			f.close()
	'''
	for i in generalZhTextProcess(text_iter,cutter=cutter,keep_punct=keep_punct,tosimplified=tosimplified,stopwords=stopwords,keep_others=keep_others,join=u'\t'):
		yield i+u'\n'
def zhWikiProcess(zhwikiin,cutter=jieba.cut,keep_punct=True,tosimplified=toSimplified,stopwords=None,min_length=1,keep_others=False,join=None):
	'''
		process a zh-wiki doc,the wiki is opened by gensim
		:param zhwikiin:
			a WikiCorpus opened by gensim.corpora.WikiCorpus,see the example
	'''
	return generalZhTextProcess(zhwikiin.get_texts(),cutter=cutter,keep_punct=keep_punct,tosimplified=tosimplified,stopwords=stopwords,min_length=min_length,keep_others=keep_others)

def rawPassagesProcess(files_iter,cutter=jieba.cut,keep_punct=True,tosimplified=toSimplified,stopwords=None,min_length=1,keep_others=False,join=None,encoding='utf8'):
	'''
		provides the same function that zhWikiProcess does, but specialized to a raw text.
		A raw text contains a plain text such as :
			俄罗斯总统普京访华
		据新华社昨日报道....
		:param files_iter:
			a file iterator or a directory containing all the raw passages that need to be processed as unit
	'''
	open_fs = map( lambda f:open(f,'rt') ,files_iter)
	for line in generalZhTextProcess(open_fs,cutter,keep_punct,tosimplified,stopwords,min_length,keep_others,join,encoding=encoding):
		yield line
	for openf in open_fs:
		openf.close()

def generalZhTextProcess_LineTester(line,cutter=jieba.cut,keep_punct=True,tosimplified=toSimplified,stopwords=None,min_length=1,keep_others=False,join=None,encoding='utf8',number_transfer=None):
	'''
		provide line level verification,for DEBUG only.
		:param line:
			a line of whitespace separated words or single word
		:return
			a processed list of word list
		Example:
			generalZhTextProcess_LineTester('Who ami',...)
	'''
	text_iter = [ [ line] ]	
	print u' '.join(list(generalZhTextProcess(text_iter,cutter,keep_punct,toSimplified,stopwords,min_length,keep_others,join,encoding,number_transfer))[0])

def generalZhTextProcess(text_iter,cutter=jieba.cut,keep_punct=True,tosimplified=toSimplified,stopwords=None,min_length=1,keep_others=False,join=None,encoding='utf8',number_transfer=None):
        '''
                process a zh text doc
                :param keep_punct:
                        keep Chinese & English punctuations or not
                :param min_length:
                        the minimum length of a sentence to be preserved
                :param text_iter:
			a text iterator that in each iteration it can return a text.Simply,you can use wiki.get_texts() or a set of raw passage file object(i.e. a value open returned) that can yield lines
			NOTE: text_iter must return a unicode string
                :param cutter:
                        chinese segment tool
                :param tosimplified:
                        if there exists some traditional characters,this is used to transfer it to simplified chinese
                        set this to None will cause no transformation, also, you can set it to return the same word,as if it does nothing.
                :param stopwords:
                        the stopwords that need to be ignored.None means nothing to ignore
                :param keep_others:
                        if a word contains character without any chinese characters, this is used to decide whether to keep them.
                        for example, a sentence like 'I have seen 一个中国人 there' will be cut to 'I',' ','have','seen'    ,'一个','中国','人',' ','there'. Intuitively, word like 'I',' ' makes no contribution to the context of the current arti    cle, they shoud be ignored
                :params join:
                        if type(join)==str,return a joined string
                        if join is None,return a vector
		:param encoding:
			the encoding used to decode a line of string.default is utf8
		:param number_transfer:
			when number is encountered, this defines how to transfer them.None no transferring.
			For example, you can define a transfer: 0 -> 数字  12 -> 数量	or in general just 数字
                :return
                        this is a generator for each result denoted by param join.A list of words or a sentence
                example:
                        from gensim.corpora import WikiCorpus
                        import utils
                        f='path/to/zhwiki-latest-pages-articles.xml.bz2'
                        wiki=WikiCorpus(f,lemmatize=False,dictionary={})
                        g=utils.generalZhTextProcess(wiki.get_texts(),cutter=utils.pynlpirCutter,stopwords=None,keep_others=False,join='\t')
                        print('\n'.join(utils.getFirstN(g,50)))
		BUGs:
			keep_others not correct, numbers should be kept, for some reason.
        '''
        if cutter is None:
                print 'cutter cannot be None'
        else:
		if join and type(join) != type(u''):
			join = join.decode('utf8')
                nonStopFilter=functools.partial(nonStopwordsFilter,stopwords=stopwords)
		if keep_others and keep_punct:
			hasChineseFilter = trueFilter
		elif keep_others and (not keep_punct):
			hasChineseFilter = lambda x:not chinesePunctFilter(x)
		elif (not keep_others) and keep_punct:
			print 'here'
                        hasChineseFilter=lambda x:chineseWordFilter(x) or chinesePunctFilter(x)
		else:
			hasChineseFilter = chineseWordFilter
	
                sent_transfer = tosimplified or originalGetter
		word_transfer = number_transfer or originalGetter
		encoding_err = 0
                for text in text_iter:
                        for sent in text:
				if type(sent)!=type(u''):
					try:
						sent=sent.decode(encoding)
					except:
						try:
							sent=sent.decode('utf8')
						except Exception,e:
							#print e
							#print 'sent is ',sent
							encoding_err += 1
							continue
                                sent=sent_transfer(sent)
                                words=cutter(sent)
                                words=filter(hasChineseFilter,words)
                                words=filter(nonStopFilter,words)
				words=map(word_transfer,words)

                                ##DEBUG
                                #print('sentence',sent)
                                #print('len of words',len(words))
                                #print('words',' '.join(words))
                                #raw_input()
                                if len(words) < min_length:
                                        continue
                                if join is None: #return Raw data
                                        yield words
                                else:
                                        yield join.join(words)
		print('Total encoding error:{}'.format(encoding_err))
def transferNumberDefault(word):
	'''
		transfer a number to uniformed word
	'''
	return u'数字' if digitsRegex.match(word) else word

def getFirstNGenerator(gen,n):
	'''
		a limited generator but still a generator
	'''
	count=0
	for i in gen:
		if count==n:
			break
		count+=1
		yield i
def getFirstN(gen,n):
	'''
		get at most n elements of the generator
	'''
	return list(getFirstNGenerator(gen,n))

def getRegexes(key):
	'''
		keys:
			zh:  contains at least one chinese character
			pure-zh:
	'''
	return regexMap[key]
def originalGetter(word):
	return word
def trueFilter(word):
	return True
def getAFilter(ffilter,*args):
	'''
		specialize a filter with given arguments
		return : a specialized filter, canonical for global generator filter that takes one argument.
		for example:
			getAFilter(nonStopWordFilter,stopwords=[',',' ','?'])
	'''
	return lambda word:ffilter(word,*args)
def nonStopwordsFilter(word,stopwords=None):
	'''
		filter, remove stop words, can be used in filter
	'''
	if stopwords is not None:
		return word not in stopwords
	return True
def chineseWordFilter(word):
	'''
		a filter used in filter
	'''
	return chRegex.match(word) is not None or digitsRegex.match(word) is not None
def chinesePunctFilter(word):
	'''
		return if a word is a punctuation
	'''
	return len(word)==1 and (chPunctRegex.match(word) is not None)
def getCommonStopwords():
	return {
		u'的',u'了',u'也',u'于',u'在',u'我',
	}
def loadStopwords(filename = None):
	'''
		load a stopwords set from a file, while is splitted by common \s characters
		filename may be any supported compressed type,such as bz2,zip or xz
		:return
			utf8 decoded stop words list
	'''
	if filename is None:
		filename = commonStopwordsFile
	ftype,fkey,fopen=guessFileType(filename)
	f=fopen(filename)
	if fkey is None:#default
		removeWindowsPreamble(f)
	stopwords=set()
	for line in f:
		for word in line.split():
			stopwords.add(word.decode('utf8'))
	f.close()
	return stopwords
def removeWindowsPreamble(f):
	'''
		windows may add three words before saving text, this is used to remove that.
	'''
	if f.read(3)!='\xef\xbb\xbf':
		f.seek(0)

def guessFileType(filename):
	'''
		may return 
			xz
			bz2
			xml
			txt
		return a type name ,the key used and a special opener
	'''
	key=os.path.splitext(filename)[1]
	if key not in openerMap:
		return key,None,openerMap[None]
	else:
		return key,key,openerMap[key]

def _initAllVariables():
	'''
		currently no lazy loading
	'''
	global regexMap
	global chRegex
	global digitsRegex
	global openerMap
	global chPunctRegex
	global commonStopwordsFile
	global shawsDataDir
	githubsDir = '/media/sf_D_DRIVE/installed/githubs'
	shawsDataDir=os.path.join(githubsDir,'shaw_s_data_respository')
	commonStopwordsFile = os.path.join(shawsDataDir,'NLP','stopword.txt')
	regexMap={'zh':u'.*[\u4e00-\u9fa5].*','digits':u'\d+','punct':u'[,., . ; : ? ! 。 ， ； ：？ ！]'}
	chRegex=re.compile(getRegexes('zh'))
	digitsRegex=re.compile(getRegexes('digits'))
	chPunctRegex=re.compile(u'''[。，、；：？！“”（）—《》‘.,;:?!"()-_<>]''')
	openerMap={'bz2':bz2.BZ2File,
			None:open,#default
		}
	if pynlpir is not None:
		pynlpir.open()

class AbstractPassageToLineSentence(object):
	'''
		process a range of different files,then return each line that can be saved into a LineSentence file
	'''
	def __iter__(self):
		#return self.lineGenerator
		pass
	def __next__(self):
		pass
class RawPassageToLineSentence(AbstractPassageToLineSentence):
	pass
class ZhWikiToLineSentence(AbstractPassageToLineSentence):
	pass

_initAllVariables()

if __name__=='__main__':
	pass
