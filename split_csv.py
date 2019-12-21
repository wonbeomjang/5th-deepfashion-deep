import os

label_dir = os.path.join('dataset', 'labels')

color_file_name = 'color.csv'
style_file_name = 'style.csv'
part_file_name = 'part.csv'
season_file_name = 'season.csv'
category_file_name = 'category.csv'

input_file_name = os.path.join(label_dir, 'labels.csv')

if not os.path.exists(label_dir):
    os.makedirs(label_dir)

input_file = open(input_file_name, 'r')
color_file = open(os.path.join(label_dir, color_file_name), 'w')
style_file = open(os.path.join(label_dir, style_file_name), 'w')
part_file = open(os.path.join(label_dir, part_file_name), 'w')
season_file = open(os.path.join(label_dir, season_file_name), 'w')
category_file = open(os.path.join(label_dir, category_file_name), 'w')

color_set = set()
style_set = set()
part_set = set()
season_set = set()
category_set = set()

while True:
    line = input_file.readline()
    if not line:
        break

    image_name, color, style, part, season, category = line.split(',')
    category = category[:-1]

    color_set.add(color)
    style_set.add(style)
    part_set.add(part)
    season_set.add(season)
    category_set.add(category)

string = ''
for color in color_set:
    string += f'{color},'
color_file.write(string[:-1])
print('color', len(string.split(',')))

string = ''
for style in style_set:
    string += f'{style},'
style_file.write(string[:-1])
print('style', len(string.split(',')))

string = ''
for part in part_set:
    string += f'{part},'
part_file.write(string[:-1])
print('part', len(string.split(',')))

string = ''
for season in season_set:
    string += f'{season},'
season_file.write(string[:-1])
print('season', len(string.split(',')))

string = ''
for category in category_set:
    string += f'{category},'
category_file.write(string[:-1])
print('category', len(string.split(',')))
