# This file is to provide basic functionality to extract results from the experimental output files


def is_character(ch):
    return ch.isalpha() or ch == '_'


def find_whole_word_match(line, keyword):
    """Find the first index in line that matches the keyword.
    If there are multiple matches, return the first one.
    The match here is the whole word match, meaning that in between the
    matching substring there are no characters, but only digits or commas.
    The whole word match is to prevent the matching substring is the substring of
    a word in line"""
    p = 0
    while p < len(line):
        word_start = p == 0 or (not is_character(
            line[p-1]) and is_character(line[p]))
        if not word_start:
            p += 1
        match_pos = line.find(keyword, p)
        if match_pos == p and word_start:
            return match_pos
        elif match_pos < 0:
            return -1
        else:
            p = match_pos
    return -1


def extract_keywords_time(f, keyword):
    # Extract the time with the keyword.
    # If there are multiple occurences, we use the last one

    ret = "NaN"
    with open(f) as infile:
        for line in infile:
            pos = find_whole_word_match(line, keyword)
            if pos >= 0:
                s = line.find('=', pos)
                t = line.find('ms', s)
                ret = line[s+1: t]
    return ret


def extract_keywords_multi_time(f, keyword):
    # Similar to extract_keywords_time, but we extract all occurences into a list as the result
    ret = []
    with open(f) as infile:
        for line in infile:
            pos = find_whole_word_match(line, keyword)
            if pos >= 0:
                s = line.find('=', pos)
                t = line.find('ms', s)
                ret.append(line[s+1: t])
    return ret


def extract_keywords_count(f, keyword):
    # Extract the count corresponding to a keyword, return the first match.
    # Can handle the case when count is a decimal

    with open(f) as infile:
        for line in infile:
            pos = find_whole_word_match(line, keyword)
            if pos >= 0:
                s = line.find('=', pos)
                s += 1
                t = s
                while t < len(line):
                    if line[t].isdigit():
                        t += 1
                    elif line[t] == '.' and t+1 < len(line) and line[t+1].isdigit():
                        t += 1
                    else:
                        break
                return line[s:t]
