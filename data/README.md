# Data

The corpus is not tracked in this repository (see `.gitignore`). To run the pipeline you need to populate this directory locally:

1. **`data/plaintext/`** — one plaintext (`.txt`, UTF-8) file per text. Books of poetry should be a single file with individual poems separated by asterisks.
2. **`data/meta.csv`** — one row per text. Columns:

   | Column | Description |
   |---|---|
   | `text_id` | Integer ID starting at 1. Appears in the `text_id` column of every result CSV. |
   | `author` | Author name, as it should appear in the prompts. |
   | `title` | Title of the work. |
   | `genre` | Free-form genre tag (e.g. `novel`, `poetry`). |
   | `filename` | Filename within `data/plaintext/`. |

   Example:
   ```
   text_id,author,title,genre,filename
   1,Author Name,Book Title,novel,1_author_book.txt
   ```
