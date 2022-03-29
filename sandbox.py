### Figuring out what the max headline length is: 
headlines = [s["Headline"] for s in ds.stances]

words = [hl.split() for hl in headlines]

longest_hl = max([len(s) for s in words])

longest_hl
  # >> longest_hl: 40 words

### Figuring out what the average headline length is: 
sum([len(s) for s in words])/len(ds.stances)
  # >> 11.12647082366123 words

### Max body length:
bodies = [ds.articles[s["Body ID"]] for s in ds.stances]

bodies_word_counts = [len(b.split()) for b in bodies]

max(bodies_word_counts)
  # >> 4788 words

### Average body length:
sum(bodies_word_counts)/len(bodies)
  # >> 369.7017129592572

