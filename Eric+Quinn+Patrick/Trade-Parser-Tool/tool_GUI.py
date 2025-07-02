import PySimpleGUI as sg
from trade_parser import parse_trade

layout = [
    [sg.Text("Paste trade string:"), sg.InputText(key="–TRADES–")],
    [sg.Button("Parse"), sg.Button("Quit")],
    [sg.Multiline(size=(60,10), key="–OUTPUT–")],
]
window = sg.Window("Trade Parser", layout)

while True:
    event, vals = window.read()
    if event in (sg.WIN_CLOSED, "Quit"):
        break
    if event == "Parse":
        trade = vals["–TRADES–"]
        parsed = parse_trade(trade)
        window["–OUTPUT–"].update(parsed)
window.close()
