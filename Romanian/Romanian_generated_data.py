import random
import csv
import uuid

numbers = [str(i) for i in range(3, 101, 1)]  
adjectives = [
    "uimitoare", "șocante", "incredibile", "secrete", "ciudate", "surprinzătoare",
    "ascunse", "nemaivăzute", "revoluționare", "extraordinare", "fantastice",
    "neobișnuite", "epice", "controversate", "magice", "inspiratoare", "geniale",
    "spectaculoase", "miraculoase", "fascinante"
]
topics = [
    "trucuri", "metode", "secrete", "lucruri", "sfaturi", "idei", "tehnici",
    "descoperiri", "povești", "erori", "strategii", "soluții", "tactici",
    "ghiduri", "planuri", "inovații", "experimente", "lecții", "rețete", "exerciții"
]
subjects = [
    "pentru a-ți schimba viața", "care îți vor salva timpul",
    "pe care nimeni nu ți le spune", "care te vor face bogat",
    "pentru a slăbi rapid", "care îți cresc productivitatea",
    "pentru o sănătate perfectă", "care îți vor aduce succesul",
    "pentru a economisi bani", "care te vor face fericit",
    "pentru o carieră de succes", "care îți vor îmbunătăți relațiile",
    "pentru a-ți găsi dragostea", "care te vor face celebru",
    "pentru o piele perfectă", "care îți vor spori încrederea",
    "pentru o casă organizată", "care îți vor schimba rutina zilnică",
    "pentru a călători mai ieftin", "care te vor ajuta să dormi mai bine"
]
calls_to_action = [
    "Trebuie să vezi asta!", "Nu vei crede ce urmează!", "Descoperă acum!",
    "Află secretul!", "Citește înainte să fie prea târziu!", "Nu rata asta!",
    "Schimbă-ți viața astăzi!", "Vezi ce au descoperit experții!", "Fă asta acum!",
    "De ce toată lumea vorbește despre asta?", "Încearcă asta astăzi!",
    "Cum aștepți să afli asta?", "Grăbește-te să vezi!", "E timpul să afli!",
    "Descoperă ce ascund ei!", "Fii primul care știe!", "Asta te va uimi!",
    "Nu mai aștepta, vezi acum!", "Secretele dezvăluite aici!", "Fă pasul acum!"
]
proper_nouns = [
    "București", "Cluj", "Timișoara", "Iași", "Brașov", "Elon Musk",
    "Andra", "Smiley", "Delia", "Inna", "Horia Brenciu", "Europa",
    "România", "Hollywood", "NASA", "Google", "Facebook", "Instagram"
]
time_contexts = [
    "în această iarnă", "vara asta", "în 2025", "în fiecare dimineață",
    "în weekend", "în vacanță", "în doar 5 minute", "peste noapte",
    "în timpul liber", "înainte de culcare"
]

#templates
templates = [
    lambda: f"{random.choice(numbers)} {random.choice(adjectives)} {random.choice(topics)} {random.choice(subjects)}! {random.choice(calls_to_action)}",
    lambda: f"De ce {random.choice(topics)} {random.choice(adjectives)} te pot ajuta {random.choice(subjects)}? {random.choice(calls_to_action)}",
    lambda: f"{random.choice(numbers)} {random.choice(topics)} {random.choice(adjectives)} pe care trebuie să le știi! {random.choice(calls_to_action)}",
    lambda: f"Cum să obții {random.choice(subjects)} cu {random.choice(numbers)} {random.choice(topics)} {random.choice(adjectives)}! {random.choice(calls_to_action)}",
    lambda: f"{random.choice(adjectives).capitalize()} {random.choice(topics)} care schimbă totul! {random.choice(calls_to_action)}",
    lambda: f"{random.choice(proper_nouns)} dezvăluie {random.choice(numbers)} {random.choice(topics)} {random.choice(adjectives)}! {random.choice(calls_to_action)}",
    lambda: f"{random.choice(numbers)} {random.choice(topics)} {random.choice(adjectives)} {random.choice(time_contexts)}! {random.choice(calls_to_action)}",
    lambda: f"Cum {random.choice(proper_nouns)} a descoperit {random.choice(topics)} {random.choice(adjectives)} pentru a {random.choice(subjects)}! {random.choice(calls_to_action)}",
    lambda: f"{random.choice(topics).capitalize()} {random.choice(adjectives)} pe care {random.choice(proper_nouns)} le folosește {random.choice(time_contexts)}! {random.choice(calls_to_action)}",
    lambda: f"De ce {random.choice(numbers)} {random.choice(topics)} {random.choice(adjectives)} sunt cheia pentru {random.choice(subjects)}? {random.choice(calls_to_action)}"
]

def generate_clickbait_headline():
    return random.choice(templates)()

headlines_data = []
generated_headlines = set()
while len(headlines_data) < 5000:
    headline = generate_clickbait_headline()
    if headline not in generated_headlines:
        generated_headlines.add(headline)
        headlines_data.append({"headline": headline, "label": 1})

output_file = "clickbait_headlines_romanian_varied.csv"
with open(output_file, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["headline", "label"])
    writer.writeheader()
    for data in headlines_data:
        writer.writerow(data)

print(f"Generated 5000 unique clickbait headlines in Romanian and saved to {output_file}")
