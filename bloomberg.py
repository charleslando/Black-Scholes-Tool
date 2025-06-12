# def get_bloomberg_data(ticker, field, start_date, end_date):
#     session = blpapi.Session()
#     if not session.start():
#         raise Exception("Failed to start Bloomberg session.")
#     if not session.openService("//blp/refdata"):
#         raise Exception("Failed to open Bloomberg service.")
#
#     ref_data_service = session.getService("//blp/refdata")
#     request = ref_data_service.createRequest("HistoricalDataRequest")
#     request.append("securities", ticker)
#     request.append("fields", field)
#     request.set("startDate", start_date.strftime("%Y%m%d"))
#     request.set("endDate", end_date.strftime("%Y%m%d"))
#
#     session.sendRequest(request)
#
#     data = []
#     while True:
#         event = session.nextEvent()
#         for msg in event:
#             if msg.hasElement("securityData"):
#                 security_data = msg.getElement("securityData")
#                 for i in range(security_data.numValues()):
#                     data.append(security_data.getValueAsElement(i))
#         if event.eventType() == blpapi.Event.RESPONSE:
#             break
#
#     return data