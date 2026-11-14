"""Excepciones de dominio para la aplicación de reconocimiento facial."""


class FaceAppError(Exception):
    """Error base de la aplicación de reconocimiento facial."""


class ConsentRequiredError(FaceAppError):
    """Se intentó registrar o procesar una persona sin consentimiento explícito válido."""


class IdentityNotFoundError(FaceAppError):
    """No existe una identidad con el identificador solicitado."""


class IdentityWithdrawnError(FaceAppError):
    """La identidad retiró su consentimiento y no puede recibir nuevas muestras."""


class NoFaceDetectedError(FaceAppError):
    """No se detectó ningún rostro en la imagen proporcionada."""


class MultipleFacesDetectedError(FaceAppError):
    """Se detectó más de un rostro cuando se exigía un único rostro."""


class DBMSClientError(FaceAppError):
    """Error al comunicarse con la API REST del DBMS subyacente."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code
