import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re
from dataclasses import dataclass
from typing import List, Optional
import pyperclip


@dataclass
class Party:
    type: str  # 'BUYER' or 'SELLER'
    name: str
    quantity: int
    details: List[str]
    clearing: str
    initiated: str


@dataclass
class Trade:
    date: str
    instrument: str
    buyers: List[Party]
    sellers: List[Party]
    price: Optional[float]
    quantity: Optional[int]
    raw_line: str


class TradeParser:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trade String Parser")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Trade String Parser",
                                font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Input section
        input_label = ttk.Label(main_frame, text="Paste your trade string here:",
                                font=('Arial', 11, 'bold'))
        input_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

        self.input_text = scrolledtext.ScrolledText(main_frame, height=15, width=80,
                                                    font=('Consolas', 10))
        self.input_text.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S),
                             pady=(0, 20))

        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))

        # Buttons
        self.parse_btn = ttk.Button(button_frame, text="Parse Trades",
                                    command=self.parse_trades, style='Accent.TButton')
        self.parse_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_btn = ttk.Button(button_frame, text="Clear All",
                                    command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.copy_btn = ttk.Button(button_frame, text="Copy Results",
                                   command=self.copy_results)
        self.copy_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.export_btn = ttk.Button(button_frame, text="Export to File",
                                     command=self.export_to_file)
        self.export_btn.pack(side=tk.LEFT)

        # Results section
        results_label = ttk.Label(main_frame, text="Results:",
                                  font=('Arial', 11, 'bold'))
        results_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))

        self.results_text = scrolledtext.ScrolledText(main_frame, height=20, width=80,
                                                      font=('Consolas', 9), state=tk.DISABLED)
        self.results_text.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

    def parse_trades(self):
        input_data = self.input_text.get("1.0", tk.END).strip()

        if not input_data:
            messagebox.showwarning("Warning", "Please paste a trade string first")
            return

        try:
            parsed_data = self.parse_trade_string(input_data)
            self.display_results(parsed_data)
        except Exception as e:
            messagebox.showerror("Error", f"Error parsing trades: {str(e)}")

    def parse_trade_string(self, input_data):
        lines = [line.strip() for line in input_data.split('\n') if line.strip()]
        trades = []
        current_trade = None
        current_date = ''
        total_lots = 0

        i = 0
        while i < len(lines):
            line = lines[i]

            # Parse master confirms date
            if 'MASTER CONFIRMS:' in line:
                current_date = line.replace('**', '').replace('MASTER CONFIRMS:', '').strip()
                i += 1
                continue

            # Parse lot total
            if 'Lot Total:' in line:
                lot_match = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)x', line)
                if lot_match:
                    total_lots += int(lot_match.group(1).replace(',', ''))
                i += 1
                continue

            # Parse trade instrument line
            if 'TRADES' in line and 'BUYER:' not in line and 'SELLER:' not in line:
                if current_trade:
                    trades.append(current_trade)

                current_trade = Trade(
                    date=current_date,
                    instrument=line.replace('**', '').strip(),
                    buyers=[],
                    sellers=[],
                    price=self.extract_price(line),
                    quantity=self.extract_quantity(line),
                    raw_line=line
                )
                i += 1
                continue

            # Parse buyer/seller lines
            if 'BUYER:' in line or 'SELLER:' in line:
                if current_trade:
                    party, next_i = self.parse_party(line, lines, i)
                    if 'BUYER:' in line:
                        current_trade.buyers.append(party)
                    else:
                        current_trade.sellers.append(party)
                    i = next_i
                    continue

            i += 1

        if current_trade:
            trades.append(current_trade)

        return {'trades': trades, 'total_lots': total_lots}

    def extract_price(self, line):
        price_match = re.search(r'TRADES\s+([\d.]+)', line)
        return float(price_match.group(1)) if price_match else None

    def extract_quantity(self, line):
        qty_match = re.search(r'\((\d+(?:,\d+)*|\d+)x\)', line)
        return int(qty_match.group(1).replace(',', '')) if qty_match else None

    def parse_party(self, line, lines, start_index):
        party = Party(
            type='BUYER' if 'BUYER:' in line else 'SELLER',
            name='',
            quantity=0,
            details=[],
            clearing='',
            initiated=''
        )

        # Extract party name and quantity from the main line
        clean_line = line.replace('**', '').replace('BUYER:', '').replace('SELLER:', '').strip()
        parts = clean_line.split()

        # Find quantity (look for pattern like "100x")
        qty_found = False
        for i, part in enumerate(parts):
            if 'x' in part and not qty_found:
                qty_match = re.search(r'(\d+)x', part)
                if qty_match:
                    party.quantity = int(qty_match.group(1))
                    party.name = ' '.join(parts[:i])
                    qty_found = True
                    break

        # Check for initiated chat
        if 'initiated' in line:
            initiated_match = re.search(r'initiated\s+(\w+\s+\w+)', line)
            if initiated_match:
                party.initiated = initiated_match.group(1)

        # Look for confirmation details in subsequent lines
        i = start_index + 1
        while i < len(lines):
            next_line = lines[i]

            if 'BUYER:' in next_line or 'SELLER:' in next_line or 'TRADES' in next_line:
                break

            if 'To Confirm:' in next_line:
                party.clearing = next_line.replace('To Confirm:', '').strip()
                i += 1
                continue

            if 'Buys' in next_line or 'Sells' in next_line:
                party.details.append(next_line.strip())

            i += 1

        return party, i

    def display_results(self, parsed_data):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)

        # Summary
        total_parties = sum(len(trade.buyers) + len(trade.sellers) for trade in parsed_data['trades'])

        summary = f"""
========================================
                 SUMMARY
========================================
Total Trades: {len(parsed_data['trades'])}
Total Lots: {parsed_data['total_lots']:,}
Total Parties: {total_parties}

"""
        self.results_text.insert(tk.END, summary)

        # Individual trades
        for idx, trade in enumerate(parsed_data['trades'], 1):
            trade_info = f"""
========================================
            TRADE {idx}
========================================
Date: {trade.date or 'Not specified'}
Instrument: {trade.instrument}
Price: {trade.price or 'Not specified'}
Quantity: {f"{trade.quantity:,}x" if trade.quantity else 'Not specified'}

"""
            self.results_text.insert(tk.END, trade_info)

            # Buyers
            for buyer in trade.buyers:
                buyer_info = f"""BUYER: {buyer.name} ({buyer.quantity}x)
  Clearing: {buyer.clearing or 'Not specified'}
"""
                if buyer.initiated:
                    buyer_info += f"  Initiated: {buyer.initiated}\n"

                if buyer.details:
                    buyer_info += "  Actions:\n"
                    for detail in buyer.details:
                        buyer_info += f"    - {detail}\n"

                buyer_info += "\n"
                self.results_text.insert(tk.END, buyer_info)

            # Sellers
            for seller in trade.sellers:
                seller_info = f"""SELLER: {seller.name} ({seller.quantity}x)
  Clearing: {seller.clearing or 'Not specified'}
"""
                if seller.initiated:
                    seller_info += f"  Initiated: {seller.initiated}\n"

                if seller.details:
                    seller_info += "  Actions:\n"
                    for detail in seller.details:
                        seller_info += f"    - {detail}\n"

                seller_info += "\n"
                self.results_text.insert(tk.END, seller_info)

        self.results_text.config(state=tk.DISABLED)

    def clear_all(self):
        self.input_text.delete("1.0", tk.END)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "Paste a trade string above and click 'Parse Trades' to see results")
        self.results_text.config(state=tk.DISABLED)

    def copy_results(self):
        results = self.results_text.get("1.0", tk.END).strip()
        if not results or "Paste a trade string" in results:
            messagebox.showwarning("Warning", "No results to copy. Please parse some trades first.")
            return

        try:
            pyperclip.copy(results)
            messagebox.showinfo("Success", "Results copied to clipboard!")
        except:
            # Fallback if pyperclip not available
            self.root.clipboard_clear()
            self.root.clipboard_append(results)
            messagebox.showinfo("Success", "Results copied to clipboard!")

    def export_to_file(self):
        results = self.results_text.get("1.0", tk.END).strip()
        if not results or "Paste a trade string" in results:
            messagebox.showwarning("Warning", "No results to export. Please parse some trades first.")
            return

        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(results)
                messagebox.showinfo("Success", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export file: {str(e)}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":


    app = TradeParser()
    app.run()