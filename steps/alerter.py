# from zenml import step, pipeline
# from zenml.integrations.slack.steps.slack_alerter_post_step import slack_alerter_post_step
# from zenml.integrations.slack.alerters.slack_alerter import SlackAlerterParameters

# @step
# def slack_custom_block_step(block_message)  :
#     my_custom_block = [
#         {
#             "type": "rich_text",
#             "elements": [
#                 {
#                     "type": "rich_text_section",
#                     "elements": [
#                         {
#                             "type": "text",
#                             "text": f" 🚨 : {block_message}, visit ZenML dashboard for reports and test-suites analysis",
#                         }
#                     ]
#                 }
#             ]
# 		},
#         {
#             "type": "rich_text",
#             "elements": [
#                 {
#                     "type": "rich_text_section",
#                     "elements": [
#                         {
#                             "type": "link",
#                             "url": "http://127.0.0.1:8237/workspaces/default/pipelines/list?page=1"
#                         }
#                     ]
#                 }
#             ]
# 		},
# 	]
#     return SlackAlerterParameters(blocks = my_custom_block)


# def warden_slackbot(message : str):
#     message_blocks = slack_custom_block_step(message)
#     post_message = slack_alerter_post_step(params = message_blocks)
