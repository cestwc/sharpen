import re

def extract_bibitem_key(bibitem):
	key_regex = r'\\bibitem{([^}]*)}'
	match = re.search(key_regex, bibitem)
	if match:
		return match.group(1)
	else:
		return None
