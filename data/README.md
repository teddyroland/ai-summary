# Data

The corpus is not tracked in this repository (see `.gitignore`). To run the pipeline you need to populate this directory locally:

1. **`data/plaintext/`** — one plaintext (`.txt`, UTF-8) file per text. Books of poetry should be a single file with individual poems separated by asterisks.
2. **`data/meta.csv`** — one row per text. Columns:

   | Column | Description |
   |---|---|
   | `TEXT_ID` | Zero-padded numeric ID (e.g. `01`, `02`). Used as a prefix in passage and summary IDs. |
   | `AUTHOR` | Author name, as it should appear in the prompts. |
   | `TITLE` | Title of the work. |
   | `GENRE` | Free-form genre tag (e.g. `novel`, `poetry`). |
   | `FILENAME` | Filename within `data/plaintext/`. |

   Example:
   ```
   TEXT_ID,AUTHOR,TITLE,GENRE,FILENAME
   01,Author Name,Book Title,novel,01_author_book.txt
   ```
