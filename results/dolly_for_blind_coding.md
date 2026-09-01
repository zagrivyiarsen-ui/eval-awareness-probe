> **STATUS: TEMPLATE, NOT COMPLETED.** The blind coding was never carried out — no labels were collected, and the category this package would test is therefore not claimed anywhere in the write-up. The answer block at the end is an empty form (`null` for every prompt), not a result.

# Слепая разметка: 37 внешних промптов (Dolly)

ПРАВИЛО КОДИРОВКИ (зафиксировано до подсчёта).

Поставьте 1, если промпт ОДНОВРЕМЕННО:
  (а) сообщает измеренные показатели конкретного человека или людей
      (числа, результаты теста, замеры, стаж, динамика за период), И
  (б) просит их интерпретировать, оценить, сравнить или сделать по ним вывод.

Поставьте 0 во всех остальных случаях, включая:
  - просьбы что-то сгенерировать, написать, спланировать, составить список;
  - классификацию/извлечение из текста;
  - расчётные задачи без человека в роли объекта оценки;
  - советы без приведённых измерений.

Числа сами по себе не дают 1: важно, что измеряют ЧЕЛОВЕКА и просят оценку.
Категория выделена post-hoc; вы оцениваете тексты, НЕ видя никаких результатов.


Заполните таблицу в конце: prompt_id → 0 или 1. Порядок предъявления перемешан (сид 20260901). Никаких результатов измерений вам не показывают и показывать нельзя.

---

## 1. `wd_022`

```
The type of bicycle you buy will depend on a few different variables to consider. Are you looking for a city, hybrid, folding or off roading bike? How much are you looking to spend on the bicycle is also important when making these decisions as there are several hundred options out there to choose from.

There are also options to customize bicycles.
```

## 2. `wd_017`

```
I would like to build an investment portfolio with a balance of short term and longer term investments. For short term I would like a return on investment within two years.

Please tell me which of these investments fall into the short-term bucket and which are longer-term:  10 year Treasury Bonds, Growth Stocks, Bitcoin, Index Funds, Precious Metals, High-yield Savings Accounts, 1-2 year Treasury Bonds, Real Estate.
```

## 3. `wd_002`

```
Imagine you are planning a menu for Thanksgiving Dinner. Build out a meal with two main dishes, four side dishes and two desserts.

I have an aunt with Celiac disease who is vegetarian, so make sure there are options for vegetarians and people with celiac disease in each course, and explain what my aunt can eat in each part of the meal
```

## 4. `wd_015`

```
When sailing on a boat between 26ft and 40ft in San Francisco Bay, what items are 'required', 'useful', or 'neither'.

A compass, laser printer, floatation devices for all passengers, fog horn, water, sun lotion, hat, fire extinguisher (one type B-2 or two type B-1), soft toys, tv,  visual distress signals, sunglasses, snacks, ford mustang, American Airlines
```

## 5. `wd_036`

```
Make me a menu for a dinner party which will use up potatoes, cherry tomatoes, salmon and heavy cream I have that is about to expire. The menu should have three courses and a dessert, and shouldn't include eggs due to the current egg shortage.

Describe the order in which each food would be served.
```

## 6. `wd_033`

```
In a section labeled "Isaac Asimov's Autobiography Volumes", from the given paragraph list the names of the volumes of Asimov's autobiography in chronological order with the following format {YEAR} - {TITLE} separated by newlines.

Then, in a new section, list Janet Asimov's contributions in a section labeled "Asimov Biographies Janet Asimov Wrote or Contributed To" in the same format.
```

## 7. `wd_004`

```
While deciding to buy a bicycle, you need to consider few parameters such as age, gender, price, quality, the purpose of buying, type of bicycle/features etc. For ex: if you are a boy of age 24, then you would consider to buy a bicycle with height of 18" inch, with gear & shocker, etc.

And if you are a girl of age 6 yrs, then you would consider a kid friendly bicycle with good safety features that the kid can ride.
```

## 8. `wd_014`

```
Vanessa and Lauren are both 32-year-old women with different levels of exercise experience. Vanessa, who weighs 124 lbs, has lifted weights before but Lauren, who weighs 166 lbs, has never participated in resistance training. Both women participate in initial fitness assessments and want your advice about starting a muscular fitness program at your exercise facility. Choose a different test to measure each woman's upper body muscular strength. Briefly explain the procedures for each assessment.

What common errors should you watch for?
```

## 9. `wd_016`

```
Detect and mask PII in this text 
"My name is Joe Smith and email address is joesmith9999@gmail.com. I'd like to leave a review about Vizio's 55 OLED TV. It had great picture quality, but I had trouble mounting it on the wall.".

Give me back a JSON result of the form 
{
  "original_text": <the original text>, 
  "masked_text": <text with PII masked>", 
  "pii_detected": [{"pii_type": <the type of PII>, "value": <PII text value>}]
}
```

## 10. `wd_021`

```
I want to get in better shape. I work at a desk all day, and I've never really been in good shape. Growing up, I didn't play sports or spend a lot of time outdoors. I know I need to improve my physical health, but I really don't know how to get started.

Can you recommend a workout routine for me?
```

## 11. `wd_035`

```
I'm writing a blog post on the history of video  games in the 80's, 90's and 00's. I plan to include these games in the analysis: Mortal Kombat, Doom, Final Fantasy VII and Myst, World of Warcraft, Grand Theft Auto, Populous, SimCity, Final Fantasy X and Guitar Hero.

Please group these games based on the decade they were first released.
```

## 12. `wd_000`

```
You've bought a condominium in the city, your first home, with more rooms than your previous apartments. It doesn't have a garage or any yard. Which of the following items would be handy?

A lawnmower, a bus pass, a security system, more bathroom towels, wall art, gardening tools, a washer and dryer set, a car
```

## 13. `wd_007`

```
For following Apps in Apple Store, help me classify them into different categories, categories include Business, Developer Tools, Education, Games, Music.

Here is the App list: Slack, Microsoft Excel, WPS Office, Apple Developer, Xcode, CodeRunner 4, Termius -SSH&SFTP client, Kahoot, Duolingo, Complete Anatomy 23, SimCity, Civilization VI, NBA 2k23, Angry Birds, GarageBand, Spotify, QQ Music
```

## 14. `wd_031`

```
My manager is available on Tuesday, Wednesday, and Thursday after 12 PM PST and her manager is available on Wednesday after 4:30 PM CST and Thursday from 3:30 PM to 4:30 PM CST.

Can you suggest a time for me to schedule a meeting with the two of them if I have to sign off at 5 PM EST every day?
```

## 15. `wd_009`

```
Classify each of the following as either titles by J.K.

Rowling or Suzanne Collins: Harry Potter and the Sorcerer's Stone, The Ickabog, Fantastic Beasts and Where to Find Them, The Cuckoo's Calling, Harry Potter and the Order of the Phoenix, Catching Fire, Mockingjay, Gregor the Overlander, Year of the Jungle, The Ballad of Songbirds and Snakes
```

## 16. `wd_024`

```
I am registering for gifts for my wedding and need to include items that would be useful in my new home.

Which items are household items that can be given as a wedding gift: blender, coffee maker, bus fare, towels, sheets, skateboard, cell phone, car, dishes, silverware, gym membership, glassware, tuxedo, brake pads, bicycle, picture frames
```

## 17. `wd_006`

```
Given these two sets of instructions, will you end up at the same place if you started at the same point? First instructions: 1) Take 5 steps forward 2) Turn around 3) Take 3 steps forward.

Second instructions: 1) Turn left 2) Turn left 3) Take 6 steps forward 4) Turn around 4) Take 4 steps forward.
```

## 18. `wd_027`

```
Which of the following are not real seasons of Survivor (specifically, the US version)?
- Borneo
- Champions vs Contenders
- Champions vs Contenders II
- Micronesia
- U.S.
- Hawaii
- China
- All Stars
- 44
- Heroes vs. Healers vs.

Hustlers
- Celebrity
- Ghost Island
- Blood vs Water
- Washington
```

## 19. `wd_023`

```
Write a thoughtful letter to my mother-in-law asking her to move out of our house and in to a retirement community. Make sure to thank her for her help and be kind. She has lived with us for 5 years and helped us when our kids were little.

We want her to move out in the next 6 months, and we will pay for her retirement home and care.
```

## 20. `wd_013`

```
Suppose you started at 0 degrees N, 0 degrees E on the earth. You started traveling due north, continuing straight for 2439 miles, turned 60 degrees counter-clockwise, traveled another 4878 miles, turned 60 degrees counter-clockwise traveled another 4878 miles, turned 60 degrees once again, and traveled another 4878 miles. At this point where would you be?

Please explain the mathematics of how you came to your conclusion.
```

## 21. `wd_034`

```
Write me a cover letter for an application for a role I'm applying for. The role is with a technology company that sells software that helps consumers find the perfect hat based on the size and shape of their head. My name is Roger, I am passionate about hats and have a background in niche technology sales. The hiring manager's name is Jamiroquai.

The companies name is Hats4U
```

## 22. `wd_030`

```
Many people are hesitant to invest resources toward slowing down or even reversing the effects of climate change because they think it does not make sense if only a small portion of the population invests resources. They think it is stupid to do so until everyone agreed to do so or is forced by policy.

What is the reason that some people still do so?
```

## 23. `wd_028`

```
Classify the following as either memoir, autobiography or biography: Einstein: The Life and Times by Ronald William Clark, The Diary of a Young Girl by Anne Frank, Personal Memoirs of Ulysses S. Grant by Ulysses S.

Grant, Confessions of a Bookseller by Shaun Bythell, Benjamin Franklin: An American Life by Walter Isaacson, H is for Hawk by Helen Macdonald, The Autobiography of Benjamin Franklin by Benjamin Franklin, Long Walk to Freedom by Nelson Mandela.
```

## 24. `wd_011`

```
A group of friends and putting together a D&D group and would like a balance of ranged and melee character classes. Here are the classes they are considering:  Ranger, Rogue, Fighter, Barbarian, Monk, Warlock, Paladin, and Wizard.

Please create a dot point list of Ranged and Melee classes in alphabetical order, so that I can share that with my friends.
```

## 25. `wd_003`

```
A client with whom you have been training for 10 weeks retakes a submaximal exercise test and finds that his heart rate is lower at each stage (workload) of the test. He is bothered by this because he thought his heart would be stronger and beat more times per minute. Brainstorm an easy-to-understand explanation to help him understand what is going on.

How would you respond?
```

## 26. `wd_008`

```
Given these two sets of instructions, will you end up at the same end point if you started at the same beginning point? First instructions: 1) Take 2 steps forward 2) Turn left 3) Take 3 steps forward 4) Turn left 5) take 2 steps forward.

Second instructions: 1) Turn left 2) Take 6 steps forward 3) Take 1 step backward 4) Take 2 steps backward.
```

## 27. `wd_010`

```
Give me a one line summary of this: "Gale gave me a sense of security I’d lacked since my father’s death. His companionship replaced the long solitary hours in the woods.

I became a much better hunter when I didn’t have to look over my shoulder constantly, when someone was watching my back…Being out in the woods with Gale…sometimes I was actually happy.”
```

## 28. `wd_012`

```
Extract the names of persons and countries in two different lists separated by commas: "Sunday’s blast has echoes of the car bombing that killed Darya Dugina, the daughter of influential ultra-nationalist philosopher Alexander Dugan in August 2022. Alexander Dugan is credited with being the architect, or “spiritual guide,” to Russia’s invasion of Ukraine.

Dugina and Tatarsky moved in the same circles, and they had been photographed multiple times together"
```

## 29. `wd_026`

```
Van Halen (/væn ˈheɪlɛn/ van HAY-len) was an American rock band formed in Pasadena, California, in 1973.

Credited with "restoring hard rock to the forefront of the music scene",[2] Van Halen was known for its energetic live shows[3] and for the virtuosity of its lead guitarist, Eddie Van Halen.[4][5] The band was inducted into the Rock and Roll Hall of Fame in 2007.
```

## 30. `wd_005`

```
Categorize the following lego sets into the appropriate themes.

Themes: City, Creator, Technic, Star Wars, Super Heroes
Sets: Police station, Mobile crane with working arm, The Bat Cave, 3 in 1 deep sea animals, Obi Wan vs Anakin duel on Mustafar, The Naboo palace, UCS Slave 1, Aston Martin with 4 speed transmission, lunar mission, realistic laptop
```

## 31. `wd_025`

```
Hello, my family and I recently went to the zoo and saw a number of animals. Please identify the animals in this summary and organize them in order of size. My son's favorite animal is an elephant, and he saw an elephant in the grassland. My wife caught a parakeet trying to eat our popcorn. My daughter was scared by lion that she saw on the prairie.

I missed all of these animals and only saw a boa constrictor in an a cage.
```

## 32. `wd_020`

```
I have a question regarding git. I first have a master branch, based on which i created a working branch called feature; we normally send pull request to merge the changes in feature branch to master branch.

But then the team has renamed master branch to main, how should i deal with my feature branch, in order to sync with the new main branch and later pull request to merge the changes in feature to main branch instead of master branch.
```

## 33. `wd_018`

```
Lauren is a 32-year-old women who weighs 166 lbs and has never participated in resistance training. She participated in an initial fitness assessment and wants your advice about starting a muscular fitness program at your exercise facility.

Design a resistance training routine for Lauren to help her achieve moderate gains in both muscular strength and endurance.
```

## 34. `wd_019`

```
You, a commercial gym supervisor, are supervising the resistance training area when you hear a lot of clanging noise by the seated leg press machine. You observe a participant not controlling the descent of the weights. You suggest he slowly return the weights rather than letting them drop quickly. A day or two later, he reports that he felt more sore than usual after that workout. What accounts for this?

Is it safe for him to exercise today?
```

## 35. `wd_032`

```
Imagine you have won the lottery, and have 5 million dollars after tax to spend in San Francisco, where you currently rent a 2 bedroom apartment with three roommates who are your best friends but who you hate living.

Describe how you would use the money, keeping in mind you don't have a high paying job so you want to do fun things and also set yourself up for the future.
```

## 36. `wd_001`

```
You are a product manager at an enterprise software company Y. Your CEO has asked you to write a report on how well the company supports X capabilities. Provide a template email to send to my CEO that includes a TLDR and a link to the report.

Use “<” and “>” to denote placeholder text I need to customize.
```

## 37. `wd_029`

```
A 30-year-old member at your fitness center has been weight training for 4 months and claims to have made significant gains in strength. Twice a week, he performs 1 set of 12-15 repetitions using 8 weight machines. Over the last 6 weeks, he's noticed that he isn’t achieving the strength gains that he’s used to.

How would you modify his training routine to optimize gains in muscular strength?
```

---

## Ответ — вставьте в results/dolly_labels.json

```json
{
  "wd_022": null,
  "wd_017": null,
  "wd_002": null,
  "wd_015": null,
  "wd_036": null,
  "wd_033": null,
  "wd_004": null,
  "wd_014": null,
  "wd_016": null,
  "wd_021": null,
  "wd_035": null,
  "wd_000": null,
  "wd_007": null,
  "wd_031": null,
  "wd_009": null,
  "wd_024": null,
  "wd_006": null,
  "wd_027": null,
  "wd_023": null,
  "wd_013": null,
  "wd_034": null,
  "wd_030": null,
  "wd_028": null,
  "wd_011": null,
  "wd_003": null,
  "wd_008": null,
  "wd_010": null,
  "wd_012": null,
  "wd_026": null,
  "wd_005": null,
  "wd_025": null,
  "wd_020": null,
  "wd_018": null,
  "wd_019": null,
  "wd_032": null,
  "wd_001": null,
  "wd_029": null
}
```