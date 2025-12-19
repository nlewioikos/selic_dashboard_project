"""
notifier.py
===========
Sistema de notificações via WhatsApp usando Twilio.

Detecta mudanças significativas em:
- Taxa Selic corrente
- Mediana do Focus
- Distribuição de probabilidades

E envia alertas automáticos.

Autor: Finance Dashboard Team
Data: 2025-12-18
"""

import logging
from typing import Optional, Dict
from datetime import datetime
from config import (
    TWILIO_ACCOUNT_SID,
    TWILIO_AUTH_TOKEN,
    TWILIO_WHATSAPP_FROM,
    TWILIO_WHATSAPP_TO,
    ENABLE_WHATSAPP_ALERTS,
    FOCUS_CHANGE_THRESHOLD,
    SELIC_CHANGE_THRESHOLD,
    PROB_CHANGE_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Verificar se Twilio está disponível
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio não instalado. Notificações desabilitadas.")


class SelicNotifier:
    """
    Sistema de notificações de mudanças em Selic/Focus/Probabilidades.
    
    Attributes:
        twilio_client: Cliente Twilio (None se desabilitado)
        last_state: Último estado conhecido para detectar mudanças
    """

    def __init__(self):
        """Inicializa notifier com Twilio (se credenciais disponíveis)."""
        self.twilio_client = None
        self.last_state: Dict = {}

        if not ENABLE_WHATSAPP_ALERTS:
            logger.info("Notificações via WhatsApp desabilitadas em config")
            return

        if not TWILIO_AVAILABLE:
            logger.warning("Twilio não disponível. Notificações desabilitadas.")
            return

        if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
            logger.warning("Credenciais Twilio não configuradas. Notificações desabilitadas.")
            return

        try:
            self.twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            logger.info("✓ Twilio conectado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao conectar Twilio: {e}")

    def _send_whatsapp(self, message: str) -> bool:
        """
        Envia mensagem via WhatsApp através do Twilio.

        Args:
            message: Texto da mensagem

        Returns:
            bool: True se enviado com sucesso
        """
        if not self.twilio_client:
            logger.debug(f"[MOCK] WhatsApp: {message}")
            return False

        try:
            msg = self.twilio_client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                to=TWILIO_WHATSAPP_TO,
                body=message,
            )
            logger.info(f"✓ WhatsApp enviado: {msg.sid}")
            return True
        except Exception as e:
            logger.error(f"Erro ao enviar WhatsApp: {e}")
            return False

    def check_and_notify_selic_change(
        self,
        current_selic: float,
        previous_selic: Optional[float] = None,
    ) -> bool:
        """
        Detecta mudança em Selic e envia alerta se significativa.

        Args:
            current_selic: Selic corrente
            previous_selic: Selic anterior (ou None para usar último estado)

        Returns:
            bool: True se notificação enviada
        """
        if previous_selic is None:
            previous_selic = self.last_state.get("selic", current_selic)

        change = abs(current_selic - previous_selic)

        if change > SELIC_CHANGE_THRESHOLD:
            direction = "📈 SUBIU" if current_selic > previous_selic else "📉 CAIU"
            message = (
                f"🚨 ALERTA SELIC\n\n"
                f"{direction} para {current_selic:.2f}%\n"
                f"Anterior: {previous_selic:.2f}%\n"
                f"Mudança: {change:+.2f}pp\n\n"
                f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            )

            self.last_state["selic"] = current_selic
            return self._send_whatsapp(message)

        self.last_state["selic"] = current_selic
        return False

    def check_and_notify_focus_change(
        self,
        current_focus: float,
        previous_focus: Optional[float] = None,
    ) -> bool:
        """
        Detecta mudança em Focus e envia alerta se significativa.

        Args:
            current_focus: Focus mediana corrente
            previous_focus: Focus anterior (ou None para usar último estado)

        Returns:
            bool: True se notificação enviada
        """
        if previous_focus is None:
            previous_focus = self.last_state.get("focus", current_focus)

        change = abs(current_focus - previous_focus)

        if change > FOCUS_CHANGE_THRESHOLD:
            direction = "📈 SUBIU" if current_focus > previous_focus else "📉 CAIU"
            message = (
                f"📊 ALERTA FOCUS\n\n"
                f"Selic em dez/2026 {direction}\n"
                f"Novo palpite: {current_focus:.2f}%\n"
                f"Anterior: {previous_focus:.2f}%\n"
                f"Mudança: {change:+.2f}pp\n\n"
                f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            )

            self.last_state["focus"] = current_focus
            return self._send_whatsapp(message)

        self.last_state["focus"] = current_focus
        return False

    def notify_probability_shift(
        self,
        scenario: str,
        new_prob: float,
        old_prob: Optional[float] = None,
    ) -> bool:
        """
        Detecta mudança significativa em probabilidade de cenário.

        Args:
            scenario: Nome do cenário ('dovish', 'central', 'hawkish')
            new_prob: Nova probabilidade
            old_prob: Probabilidade anterior

        Returns:
            bool: True se notificação enviada
        """
        if old_prob is None:
            old_prob = self.last_state.get(f"prob_{scenario}", new_prob)

        change = abs(new_prob - old_prob)

        if change > PROB_CHANGE_THRESHOLD:
            direction = "📈 AUMENTOU" if new_prob > old_prob else "📉 DIMINUIU"
            emoji = "🕊️" if scenario == "dovish" else "⚖️" if scenario == "central" else "🦅"

            message = (
                f"{emoji} CENÁRIO {scenario.upper()}\n\n"
                f"Probabilidade {direction}\n"
                f"De {old_prob*100:.1f}% para {new_prob*100:.1f}%\n"
                f"Mudança: {change*100:+.1f}pp\n\n"
                f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            )

            self.last_state[f"prob_{scenario}"] = new_prob
            return self._send_whatsapp(message)

        self.last_state[f"prob_{scenario}"] = new_prob
        return False

    def send_daily_summary(self, summary_data: Dict) -> bool:
        """
        Envia resumo diário das projeções de Selic.

        Args:
            summary_data: Dict com dados do dia (selic, focus, probs)

        Returns:
            bool: True se enviado
        """
        message = (
            f"📈 RESUMO DIÁRIO - SELIC\n\n"
            f"💰 Selic Corrente: {summary_data.get('selic', 'N/A'):.2f}%\n"
            f"📊 Focus 2026: {summary_data.get('focus', 'N/A'):.2f}%\n"
            f"📉 IPCA 12m: {summary_data.get('ipca', 'N/A'):.2f}%\n\n"
            f"🕊️ Dovish: {summary_data.get('prob_dovish', 0)*100:.1f}%\n"
            f"⚖️ Central: {summary_data.get('prob_central', 0)*100:.1f}%\n"
            f"🦅 Hawkish: {summary_data.get('prob_hawkish', 0)*100:.1f}%\n\n"
            f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )

        return self._send_whatsapp(message)


# Singleton global
_notifier = None


def get_notifier() -> SelicNotifier:
    """Retorna instância singleton do notifier."""
    global _notifier
    if _notifier is None:
        _notifier = SelicNotifier()
    return _notifier
