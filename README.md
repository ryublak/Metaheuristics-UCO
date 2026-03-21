# Metaheuristics Coursework 🚀

Central repository containing the assignments and algorithms developed for the Metaheuristics course. 

All code is written in Python, with virtual environments and dependencies managed quickly and efficiently using `uv`.

## 📁 Project Structure

The repository is organized by independent assignments. Each folder contains its own source code, data, and dependency tracking file (`uv.lock`).

* `Practice 1/`: [Brief description of practice 1, e.g., Local search algorithms]
* `Practice 2/`: [Brief description, e.g., Wine quality prediction using winequality-red.csv]
* `Practice 3/`: [To be added...]

## 🛠️ Setup and Installation

To run any of the assignments locally, ensure you have [uv](https://github.com/astral-sh/uv) installed on your system.

1. **Clone this repository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   ```

2. **Navigate to the desired assignment folder:**
   ```bash
   cd "your-repo-name/Practice 2"
   ```

3. **Install dependencies and automatically create the environment:**
   ```bash
   uv sync
   ```
   *Note: This reads the `uv.lock` file to replicate the exact environment used for the project.*

## 🚀 Usage

Once the environment is synced with `uv`, you can run the main script directly with:

```bash
uv run src/main.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
