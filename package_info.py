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


class Package(metaclass=ABCMeta):

	def __init__(self, name, url):
		self.name = name
		self.url = url
		# storing scraped web page
		self.page = BeautifulSoup(requests.get(url).text, 'lxml')

	@property
	@abstractmethod
	def author(self):
		pass

	@property
	@abstractmethod
	def description(self):
		pass

	def to_dict(self):
		fields = get_properties(self.__class__) + ['url']
		return {f: getattr(self, f) for f in fields}


class GithubPackage(Package):

	def __init__(self, name, url):
		if url.find('github.com') == -1:
			raise ValueError('The url suggests that the package is not hosted on Github.')
		super().__init__(name, url)

	@property
	def author(self):
		return self.page.find(attrs={'rel': 'author'}).text

	@property
	def description(self):
		return self.page.find(attrs={'itemprop': 'about'}).text.strip().capitalize()

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

	cols = df.columns
	df_h = pd.DataFrame([['---',]*len(cols)], columns=cols)
	df = pd.concat([df_h, df])
	df.to_csv(s, sep='|', index=False)
	return s.getvalue()

def update_readme(info, template_path='./README.template'):
	with open(template_path, 'r') as f:
		template = f.read()
	print(template)
	template = template.replace(r'{{ tbl }}', convert_info_to_markdown(info))
	print(template)
	with open('./README.md', 'w', encoding='utf8') as f:
		f.write(template)

if __name__ == '__main__':
	pkgs = list_packages()
	info = get_existing_info()

	for name, url in pkgs:
		if name not in info:
			pkg = make_package(name, url)
			info[pkg.name] = pkg.to_dict()

	update_readme(info)