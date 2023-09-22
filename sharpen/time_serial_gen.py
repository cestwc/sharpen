import datetime

CHARACTERS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def generate_serial():
	now = datetime.datetime.now()
	combined_number = now.year % 10 * 1000000 + now.month * 10000 + now.day * 100 + now.hour
	return int_to_baseX(combined_number, len(CHARACTERS))

def int_to_baseX(num, base):
	return CHARACTERS[num] if num == 0 else ''.join(CHARACTERS[num % base] for num in divmod_gen(num, base))

def divmod_gen(num, base):
	while num:
		num, rem = divmod(num, base)
		yield rem

# generate a unique 4-character serial number for each hour within a decade
# print(generate_serial())
