# Where the paper is

The paper is on Zenodo at https://doi.org/10.5281/zenodo.23899100, version 1, published 2026-11-02 under CC BY 4.0. The deposited file is `paper/induction-tab.pdf` at md5 `2251485b8bc84e93dc6efc2036ce551b`. https://doi.org/10.5281/zenodo.23899099 always resolves to the latest version, and a changed PDF goes up as a new version of that record.

The rest of this file is what an arXiv submission would need. Those details come from arXiv's help pages as read on 2026-11-02, so check the live pages before relying on them.

## Decisions for a person

1. **Whether to submit.** An announced arXiv paper is part of the permanent scholarly record, and arXiv only considers removal if the submitter had no legal right to agree to the license.
2. **Reporting tool use.** arXiv's moderation page requires authors to report any significant use of sophisticated tools, and it names text-to-text generative AI among them. The paper has no such statement now. If generative AI tools were used for the code, analysis or text, the author decides the wording and adds it, for example to the Credits paragraph. By signing, the author takes full responsibility for all of the contents, however they were generated.
3. **The endorser.** See the next section.
4. **The license.** arXiv calls the chosen license irrevocable, though different versions of a paper can have different licenses. The MIT license on the repository covers the code, and CC BY 4.0 covers the Zenodo record. Neither chooses a license for the arXiv version.
5. **The category.** The suggestion is cs.LG as primary. Any cross-list is the author's call, and moderators may reclassify.

## What arXiv needs

### An account and possibly an endorsement

A first-time submitter may need an endorsement for the category. An institutional email address can speed this up, and a submitter without one can seek a personal endorsement from an established arXiv author. arXiv says it is best to find someone you know personally who knows the subject area.

To ask for a personal endorsement:

1. Start a new submission and select cs.LG.
2. Look in your email for the endorsement request email. It has a link to give to a prospective endorser.
3. Open the abstract page of a recent arXiv paper this one cites, and follow "Which authors of this paper are endorsers?" at the bottom.
4. Send an eligible endorser the request email. You can include a copy of the paper.

At least one positive endorsement is required for the category. arXiv says it is inappropriate to email large numbers of potential endorsers at once, or to email the same endorser repeatedly.

### The source files

Upload these five files, as a zip or tar.gz or one by one:

- `induction-tab.tex`
- `numbers.tex`
- `fig-survival.tex`
- `fig-depth.tex`
- `fig-gap.tex`

Leave out the PDF and the Python scripts. They stay in the repository, and the Check Files step flags files that look extraneous.

The paper has only been built with tectonic, which uses XeTeX. arXiv recognizes pdflatex, and the source uses T1 fonts, Latin Modern and TikZ with no `fontspec`, so it should compile there, but that compile is untested because pdflatex is not installed here. At Check Files, note which compiler arXiv picked. Read the PDF that arXiv builds before going further, and check the page count, the three figures and Table 3's width.

### The metadata

The form asks for the title, the authors and the abstract as text. Copy the abstract from the PDF, because the source holds macros in place of its numbers. It is about 1360 characters. arXiv's limits are in its separate title and abstract instructions (`https://info.arxiv.org/help/prep.html`), which were not read for this file.

### Before you submit

- Both links under the title must resolve, because the paper says the code and result files are in that repository and the page runs at that address.
- Run `python3 paper/numbers.py`, and rebuild if it reports anything.
- Expect moderation before the paper is public. arXiv reserves the right to reclassify or decline any submission.
