# Gold span annotation — instructions for the annotator

Thank you. This takes about 45–70 minutes in total and can be done in six
sittings of ten questions. Your answers are saved in your browser as you go.

## What we are asking you to do

Our system answers a nurse's question by **pointing at a sentence in a
guideline** rather than writing a new sentence. It never composes prose: it
picks one of the passages already in the document and shows it exactly as
printed, with the page and the source.

We can already prove it never invents text. What we cannot yet tell is whether
it points at the **right** passage. That needs a nurse's judgement, which is
what this sheet collects.

For each question you see the same numbered list of passages the software
chooses from. Your job is to mark which of them you would be willing to show a
nurse who asked that question.

## How to use the sheet

Open `study_001-010.html` in a browser (double-click it). Then, per question:

1. Type your initials in the box at the top, once.
2. For each of the two fields — **action** and **quantity** — choose a verdict:
   - *the chunk states it* — then mark the passage(s) below;
   - *not stated in this chunk* — the passage you would need simply is not
     there. **This is a real and useful answer. Please use it whenever it is
     true, and do not stretch to find a near-match;**
   - *cannot tell*.
3. To mark a passage, click it:
   - **once** → green `*`, the single best passage;
   - **twice** → blue `+`, also acceptable;
   - **three times** → unmarked.

   Mark **every** passage you would be content to show a nurse, not only your
   favourite. Guidelines often say the same thing twice — once in a paragraph
   and again in a summary table — and both are correct answers.
4. Add a one-line note if something is worth saying: why nothing fits, which
   wording is misleading, or a safety concern. These notes are valuable; please
   use them.
5. When the batch is done, press **download answers** and send us the file.

## Two things that matter for the result

**Please do not open "show the written reference answer" until after you have
chosen.** That text was written by a language model from this same passage, so
reading it first tells you where the answer came from and stops you judging
whether the passage actually answers the nurse's question. The sheet records
whether it was opened, and we exclude those from the main result — so opening
it early costs us the item. If you are stuck, mark *cannot tell* instead.

**Judge the question as asked, for a general patient.** Do not assume facts not
in the question, and do not reject a passage only because it would need
adapting to a particular patient. If the passage answers the question as a
guideline statement, it counts.

## Notes on the material

A few questions have a passage in **Italian** — this corpus includes the
Italian translation of the WHO surgical site infection guideline and a small
number of questions were mislabelled against it. Please mark those *not stated
in this chunk* (or *cannot tell*) and note it. You are not expected to read
Italian, and finding these is genuinely useful to us.

Some questions have **no quantity candidate at all**; the sheet says so and
there is nothing to mark for that field. That is expected for about half of
them.

## If the sheet does not work

If you see a red error box, or the page is blank, please send us the message and
use `study_all.csv` instead — the same questions and passages in a spreadsheet,
with a `mark` column to type `best` or `ok` into.

## Files

| file | what it is |
|---|---|
| `study_001-010.html` … `study_051-060.html` | the six batches, ten questions each |
| `study_all.csv` | spreadsheet fallback, same content |
| `study_manifest.json` | which questions were sampled, and the seed |
| `README.md` | this file |

Everything stays on your computer until you send the downloaded file back.
Nothing is uploaded by the sheet itself.
