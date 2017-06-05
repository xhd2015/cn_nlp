#encoding=utf8
'''
	this file is for user to run,each function is a runnable command
'''
import sys
import utils

supported_run = [
	'run_no_thing',
	'zhRawToLineSentence',
	'zhRawPassagesToLineSentence',
	'zhWikiTransferred',
	'trainBasedOnLineSentence'
]

def run_select():
	print 'Supported funcs:'
	glb = globals()
	print '\n'.join( map(lambda x:"\t[{}] {}\t{}".format(x[0],x[1],glb[x[1]].__doc__ or 'no help message'), enumerate(supported_run)))
	sel = utils.askUntil('Please select a func to run',checker=utils.intChecker,default='0')
	sel = int(sel)
	globals()[supported_run[sel]]
	print 'run end.'
	

def run_no_thing():
	pass
def zhRawToLineSentence(f,fout):
	import utils
	line_iters=utils.rawPassagesProcess([f],keep_punct=False,join=u' ')
	with open(fout,'wt') as outf:
		for line in line_iters:
			outf.write(line.encode('u8')+'\n')
	outf.close()
def zhRawPassagesToLineSentence():
	'''
		process data that has raw format
		by default,if it encounter with some encoding error, it will pass by that sentence
	'''
	import utils
	import os
	import os.path
	public_dir = '/home/fulton/D_githubs/NLPIR/NLPIR-Parser'
	sub_dirs = ['演示语料','训练分类用文本']
	dir1=os.path.join(public_dir,sub_dirs[0])
	files = map(lambda f:os.path.join(dir1,f),os.listdir(dir1))
	dir2 = os.path.join(public_dir,sub_dirs[1])
	for subsubdir in os.listdir(dir2):
		cur_subdir=os.path.join(dir2,subsubdir)
		for subfile in os.listdir(cur_subdir):
			files.append(os.path.join(cur_subdir,subfile))
	print '\n'.join(files)
	target_outf='all.LineSentence.txt'
	print 'output :',target_outf
	line_iters = utils.rawPassagesProcess(files,keep_punct=False,join=u' ',encoding='gbk')
	with open(target_outf,'wt') as outf:
		for line in line_iters:
			outf.write(line.encode('u8')+'\n')
	'''
	NOTE:
		write only accepts string.In python2, string is not the same with python3.It's more like bytes in python3
		This fact is demonstrated by the usage of encode and decode
	'''
	outf.close()
	
	
def zhWikiTransferred():
	import utils
	bzpath=open('wiki_path.txt').read().strip()
	outpath = utils.askUntil('input a save location:',checker=lambda x:True,default=None)
	newf=utils.saveZhWikiLineSentence(bzpath,outpath)
	print('saved new LineSentence file is {}'.format(newf))

def trainBasedOnLineSentence():
	import gensim,os.path,utils
	linef=utils.askUntil('input the LineSentence file path to use:',duration=60*60)
	size = utils.askUntil('input a size of the word2vec:',checker=utils.intChecker,default='300')
	size = int(size)
	linedir=os.path.dirname(linef)
	sents = gensim.models.word2vec.LineSentence(linef)
	now = utils.getCurrentTimestamp()
	print('size={}, LineFile={},now={}'.format(size,linef,now))
	print('training...')
	model=gensim.models.Word2Vec(sents,size=size,window=5,min_count=5,workers=2,iter=3)
	model.save( os.path.join(linedir,'wiki_vec_{}_args_size={}_window=5_min_count=5_workers=2_iter=3'.format(now,size)) )


if __name__=='__main__':
	run_select()
