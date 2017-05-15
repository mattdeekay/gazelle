"""
Exceptions used in Gazelle, for debugging and stuff.

"""


class GazelleError(Exception):
	""" Base class for exceptions related to our Gazelle project.
	"""
	def __init__(self, message):
		self.message = message

class DetectionError(GazelleError):
	""" Exceptions during face/eye detection.
	"""
	pass