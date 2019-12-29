import os


def main():
    label_dir = os.path.join('dataset', 'labels')
    ios_label_dir = os.path.join('dataset', 'ios_labels')
    input_file_name = os.path.join(label_dir, 'labels.csv')

    def save_csv(category, set):
        file = open(os.path.join(label_dir, f'{category}.csv'), 'w')
        ios_file = open(os.path.join(ios_label_dir, f'{category}.csv'), 'w')

        string = ''
        for item in set:
            ios_file.write(f'{item}\n')
            string += f'{item},'
        file.write(string[:-1])

        file.close()
        ios_file.close()

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(ios_label_dir):
        os.makedirs(ios_label_dir)

    input_file = open(input_file_name, 'r')

    color_set = set()
    style_set = set()
    part_set = set()
    season_set = set()
    category_set = set()

    while True:
        line = input_file.readline()
        if not line:
            break
        if line.startswith("image_file_name,color,style,part,season,category"):
            continue

        image_name, color, style, part, season, category = line.split(',')
        category = category[:-1]

        color_set.add(color)
        style_set.add(style)
        part_set.add(part)
        season_set.add(season)
        category_set.add(category)

    input_file.close()

    save_csv('color', color_set)
    save_csv('style', style_set)
    save_csv('part', part_set)
    save_csv('season', season_set)
    save_csv('category', category_set)

    file = open(os.path.join('dataset', 'label_data.txt'), 'w')
    file.write(f'color,style,part,season,category\n')
    file.write(f'{len(color_set)},{len(style_set)},{len(part_set)},{len(season_set)},{len(category_set)}')
    file.close()


if __name__ == '__main__':
    main()