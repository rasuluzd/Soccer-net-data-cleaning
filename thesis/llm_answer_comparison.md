# LLM Answer Quality: RAW vs CLEANED ES indexes

Same retrieval (ES BM25 + AUTO fuzzy + phrase boost) feeding the
same Mistral 7B (Ollama) with identical prompt. The only variable
is whether the indexed text was the raw Whisper output or the
ASR-cleaning pipeline's output.

This is the experiment that DOES distinguish raw from cleaned —
the LLM is the downstream consumer that needs canonical entity
names to give a correct answer.

## Q1: Who scored the second goal for Liverpool?

### Question: `Who scored the second goal for Liverpool?`

**RAW Whisper context fed to Mistral:**

```
1. [56min] in his second goal Daniel Sturridge applauding the Liverpool fans and not the Chelsea majority who used to cheer him and Divock Origi
2. [46min] expect to be under some pressure Chelsea will surely be better in the second half one thing's for certain if they do get a goal back Liverpool can
3. [56min] he's struggling, he's wrestling with it he played a part in that Chelsea equaliser at Swansea didn't he by Trivian, the ball looped up for Costa in his second goal Daniel Sturridge applauding the Liverpool fans
```

**Mistral answer (RAW):**

> Daniel Sturridge scored the second goal for Liverpool. [56min] he's wrestling with it he played a part in that Chelsea equaliser at Swansea didn't he by Trivian, the ball looped up for Costa in his second goal Daniel Sturridge applauding the Liverpool fans

**CLEANED context fed to Mistral:**

```
1. [46min] Expect to be under some pressure. Chelsea will surely be better in the second half. One thing's for certain: If they do get a goal back. Liverpool can.
2. [56min] He played a part in that Chelsea equaliser at Swansea, didn't he? By Trivian, the ball looped up for Costa. In his second goal. Daniel Sturridge applauding the Liverpool fans. And not the Chelsea.
3. [37min] I'm sure. Gary. Over and over again. In the first goal. The second goal.
```

**Mistral answer (CLEANED):**

> NO_MATCH (The provided commentary segments do not specify who scored the second goal for Liverpool.)

---

## Q2: What did Diego Costa do in the first half?

### Question: `What did Diego Costa do in the first half?`

**RAW Whisper context fed to Mistral:**

```
1. [44min] this has been as good as it was exceptional from City in that period it will be better in the sense that they don't concede of course because Manchester City did in the last couple of minutes had the first half which changed the complexion of that game that didn't cost them the win
2. [32min] they're looking for finding themselves with a bit more work to do in midfield and even deeper than that here's Diego Costa getting better Chelsea
3. [29min] one good attack they had is when they did get it into Diego Costa Asa always offers them hope. Matic. Oskar.
```

**Mistral answer (RAW):**

> Diego Costa was getting better for Chelsea in the first half, as mentioned in segment 3 [32min].

**CLEANED context fed to Mistral:**

```
1. [44min] It will be better in the sense that they don't concede, of course. Because Manchester City did in the last couple of minutes. Had the first half, which changed the complexion of that game. That didn't cost them the win. Chelsea play Leicester in the EFL Cup.
2. [32min] With a bit more work to do in midfield. And even deeper than that. Here's Diego Costa. Getting better Chelsea. There's no doubt.
3. [29min] One good attack they had. Is when they did get it into Diego Costa. Asa always offers them hope. Matic. Oscar.
```

**Mistral answer (CLEANED):**

> The provided commentary segments do not explicitly state what Diego Costa did in the first half, but they imply that he was involved in one good attack by Chelsea. [32min] "With a bit more work to do in midfield. And even deeper than that. Here's Diego Costa. Getting better Chelsea."

---

## Q3: Who is the Chelsea right-back?

### Question: `Who is the Chelsea right-back?`

**RAW Whisper context fed to Mistral:**

```
1. [11min] Chelsea sit just to push them up the pitch they do look solid difficult to break down pull up on four of their last eight here Chelsea is a five across the pitch Oscar just breaking out now back in tight four and five 5-1 for Chelsea
2. [58min] wing backs here that I can see at Chelsea you might ask William but is he really
3. [51min] and the swipe at it really slid forward by Sadio Mane Klein Conte who was attacking a moment or two ago is back harassing and fouling
```

**Mistral answer (RAW):**

> NO_MATCH (The provided commentary segments do not mention the name of the Chelsea right-back.)

**CLEANED context fed to Mistral:**

```
1. [11min] Chelsea. Sit just to push them up the pitch. They do look solid, difficult to break down. Pull up on four of their last eight here. Chelsea is a five across the pitch. Oscar just breaking out, now back in tight four and five. 5-1 for Chelsea.
2. [58min] That I can see. At Chelsea. You might ask Willian. But is he really? A wing back.
3. [32min] Similar. Rush to the right back. Of Chelsea. Going away from goal. Not going anywhere.
```

**Mistral answer (CLEANED):**

> NO_MATCH (The provided commentary segments do not mention the name of the Chelsea right-back.)

---

## Q4: Tell me about Sturridge's involvement in the match.

### Question: `Tell me about Sturridge's involvement in the match.`

**RAW Whisper context fed to Mistral:**

```
1. [35min] playing on side side the touches at all he was so frustrated his involvement in this first 35 minutes or lack of involvement just say able to sort of get himself a head of steam up now in terms of his temper as well he seems to thrive on
2. [3min] he's definitely added pace since this Chelsea team constantly play at a fire heart for higher John Terry mentioning about Louise the strange thing for me that this attack play out with Lallana beautiful goal in that match to ask the least good against Leicester the winner
3. [75min] and in the end required in a simple fashion it didn't surprise me Liverpool but I don't trust them to see the game out
```

**Mistral answer (RAW):**

> Sturridge was not heavily involved in the first 35 minutes of the match, struggling to get himself a head of steam up. [1] However, his pace seemed to have increased since then, as he played a part in Liverpool's attack that resulted in a beautiful goal. [2]

**CLEANED context fed to Mistral:**

```
1. [35min] Side. The touches at all. He was so frustrated. His involvement in this first 35 minutes. Or lack of involvement, just say able to sort of get himself a head of steam up. Now, in terms of his temper as well, he seems to thrive on. Getting pumped up.
2. [3min] He's definitely added pace, since this Chelsea team constantly play at a fire. Heart for higher John Terry. Mentioning about Louise, the strange thing for me that this attack play out. With Lallana. Beautiful goal in that match, to ask the least good - against Leicester. The winner.
3. [75min] Goalkeeper will be required. And, in the end, required. In a simple fashion. It didn't surprise me Liverpool. But,
```

**Mistral answer (CLEANED):**

> Sturridge's involvement in the match was described as lacking in the first 35 minutes, with difficulty getting himself a head of steam up. However, no specific mention of his contribution to any goal or significant play was made in the provided commentary segments.

---

## Q5: Did Aspilicueta have a header?

### Question: `Did Aspilicueta have a header?`

**RAW Whisper context fed to Mistral:**

```
1. [29min] through Liverpool Lovren's header counter one good attack they had is when they did get it into Diego Costa
2. [30min] side Matic Interesting had Mane been able to control that it's a real test for him Swapped Mane and Sturridge at the moment Liverpool sorry he's got to keep an eye on Aspilicueta pushing on
3. [88min] Liverpool have had so much supremacy very good value for their 2-0 half-time lead
```

**Mistral answer (RAW):**

> NO_MATCH. The provided commentary segments do not mention any header by Aspilicueta.

**CLEANED context fed to Mistral:**

```
1. [29min] Through Liverpool. Lovren's header. Counter. One good attack they had. Is when they did get it into Diego Costa.
2. [55min] And they nearly pinched that. Chelsea would have been in trouble. Here's Azpilicueta. A little bit more fervour in the Chelsea. Support.
3. [20min] Coutinho's in. Coutinho's in. Far side. Sturridge didn't have the highs in the back of his head. Fletcher's out to the left for Mane. For Mane.
```

**Mistral answer (CLEANED):**

> NO_MATCH

---

## Q6: Who was the Liverpool goalkeeper?

### Question: `Who was the Liverpool goalkeeper?`

**RAW Whisper context fed to Mistral:**

```
1. [89min] for Wijnaldum who has been showing signs of fatigue getting a senior career with Liverpool he was with Tottenham Alex Inglethorpe moved from
2. [69min] the Chelsea the counter-attack great throw from the goalkeeper this remember Liverpool they've got the energy we're all racing back
3. [81min] yards out that's the game isn't it yeah big moments still in the balance still got his hands on his head Lucas who played as makeshift centre-back against Leicester Tino not happy to be hooked defensive change Lucas's last goal for Liverpool was six years ago this very
```

**Mistral answer (RAW):**

> NO_MATCH (The provided commentary segments do not mention the name of the Liverpool goalkeeper.)

**CLEANED context fed to Mistral:**

```
1. [89min] Bringing on Kevin Stewart. For Wijnaldum. Who has been showing signs of fatigue. Getting a senior career with Liverpool. He was with Tottenham.
2. [61min] Diego Costa, but I have to say, Liverpool's vulnerability surfacing out of nowhere there, he's alive. Get back and get on with. It was the message from the manager, it was not an easy finish, It wasn't the most straightforward of heights and he had to get in it to beat. The goalkeeper, the man on the line. Liverpool now can see the free kick. That's a completely different game. The.
3. [81min] Lucas, who played as makeshift centre-back against Leicester, not. Happy to be hooked. Defensive change. Lucas's last goal for Liverpool was six years ago. This very. Date. Chelsea have three lined up to come on, but they make a Mourinho like situation.
```

**Mistral answer (CLEANED):**

> NO_MATCH (The provided commentary segments do not mention the name of the Liverpool goalkeeper.)

---

## Q7: What happened with David Luiz?

### Question: `What happened with David Luiz?`

**RAW Whisper context fed to Mistral:**

```
1. [53min] he's not tried any Cruyff turns, he's just kicked it out of play Matip couldn't keep it in you just feel with David Luiz that Chelsea
2. [53min] you just feel with David Luiz that Chelsea would have had a list of central defenders they wanted to sign and I'm not sure how near the top of it he was
3. [90min] points to stay firm with the first ball but we read it to play short in the middle played by David Luiz Liverpool holding a line
```

**Mistral answer (RAW):**

> At the 53rd minute, there was a discussion about David Luiz not trying any Cruyff turns and later mentioning that Chelsea might have had other central defenders they preferred to sign. However, no specific event related to him was mentioned in the given commentary segments. (NO_MATCH)

**CLEANED context fed to Mistral:**

```
1. [53min] Matip. Couldn't keep it in. You just feel with David Luiz. That Chelsea. Would have had a list of central defenders they wanted.
2. [21min] Liverpool. David Luiz. David Luiz, shaken up by that bang in the face. Look at Chelsea, the static. Ivanovic stood still, Kante, they're really not mobile.
3. [60min] Are monitored to run further than any other team. In the opening round of games. David Luiz running with the ball. Matic. Now Willian.
```

**Mistral answer (CLEANED):**

> David Luiz was shaken up by a bang in the face during the match. [21min]

---
