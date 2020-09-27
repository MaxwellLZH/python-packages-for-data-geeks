from abc import ABCMeta, abstractmethod
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import json


def get_properties(obj):
	""" Get a list of properties for a given object """
	prpt = list()
	for k, v in obj.__dict__.items():
		if isinstance(v, property):
			prpt.append(k)
	return prpt


def convert_to_float(s):
	s = s.replace('k', '* 1000')
	return eval(s)


class Package(metaclass=ABCMeta):

	def __init__(self, name, url):
		self.name = name
		self.url = url
		# storing scraped web page
		self.page = BeautifulSoup(requests.get(url).text, 'lxml')

	@property
	@abstractmethod
	def owner(self):
		pass

	@property
	@abstractmethod
	def description(self):
		pass

	def to_dict(self):
		fields = get_properties(self.__class__) + ['name', 'url']
		return {f: getattr(self, f) for f in fields}


class GithubPackage(Package):

	def __init__(self, name, url):
		if url.find('github.com') == -1:
			raise ValueError('The url suggests that the package is not hosted on Github.')
		super().__init__(name, url)

	@property
	def owner(self):
		return self.page.find(attrs={'rel': 'author'}).text

	@property
	def description(self):
		des = self.page.find(class_='BorderGrid-cell').find(class_="f4 mt-3")
		# des = self.page.find(attrs={'itemprop': 'about'})
		if des is None:
			return ''
		else:
			return des.text.strip().capitalize()

	@property
	def star_count(self):
		return self.page.find('a', attrs={'class': 'social-count js-social-count'}).text.strip()


def list_packages(path='./packages.json'):
	""" Get a list of (name, url) from packages.json file """
	with open(path, 'r') as f:
		packages = json.load(f)
	res = list()
	for cat, pkgs in packages.items():
		res.extend([(p['name'], p['url']) for p in pkgs])
	return res

def get_existing_info(path='./package_info.json'):
	with open(path, 'r') as f:
		package_info = json.load(f)
	return package_info

def make_package(name, url):
	""" Create Package object according to url """
	if 'github.com' in url:
		return GithubPackage(name, url)
	else:
		raise NotImplementedError('Only supports Github package for now.')

def save_info(info, path='./package_info.json'):
	with open(path, 'w') as f:
		json.dump(info, f)

def convert_info_to_markdown(info):
	import pandas as pd
	from io import StringIO

	s = StringIO()
	df = pd.DataFrame(info).T
	
	# formatting for github package
	df = df[['name', 'owner', 'star_count', 'description', 'url']]
	df = df.rename(columns={'url': 'link', 'star_count': 'stars'})
	df['name'] = df['name'].apply(lambda x: '[**{}**]'.format(x) if x != '---' else x)
	df['link'] = '(' + df['link'] + ')'
	df['name'] = df['name'] + df['link']
	df = df.drop('link', axis=1)

	# sort packages for start count
	df['stars_float'] = df['stars'].apply(convert_to_float)
	df = df.sort_values('stars_float', ascending=False).drop('stars_float', axis=1)

	# add header mark
	cols = df.columns
	df_h = pd.DataFrame([['---',]*len(cols)], columns=cols)
	df = pd.concat([df_h, df])

	df.to_csv(s, sep='|', index=False)
	return s.getvalue()

def update_readme(info, package_path='./packages.json', template_path='./README.template'):
	from jinja2 import Template

	with open(package_path, 'r') as f:
		packages = json.load(f)

	# mapping from category to its own markdown table
	packages = {cat: {p['name']: info[p['name']] for p in pkgs} for cat, pkgs in packages.items()}
	packages = {cat: convert_info_to_markdown(info) for cat, info in packages.items()}

	with open(template_path, 'r') as f:
		template = Template(f.read())

	template = template.render(info=packages)
	with open('./README.md', 'w', encoding='utf8') as f:
		f.write(template)


if __name__ == '__main__':
	pkgs = list_packages()
	info = get_existing_info()

	# get information for new packages
	for name, url in pkgs:
		if (name == 'horovod') or (name not in info) or (info[name]['description'].strip() == ''):
			pkg = make_package(name, url)
			info[pkg.name] = pkg.to_dict()

	save_info(info)
	update_readme(info)