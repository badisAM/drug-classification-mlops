"""
Module pour envoyer les métriques et logs MLflow vers Elasticsearch
"""
import json
from datetime import datetime
from elasticsearch import Elasticsearch
from typing import Dict, Any, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ElasticsearchLogger:
    """Classe pour gérer l'envoi de logs vers Elasticsearch"""
    
    def __init__(self, hosts: list = None, index_name: str = "mlflow-metrics"):
        """
        Initialise la connexion à Elasticsearch
        
        Args:
            hosts: Liste des hôtes Elasticsearch (défaut: ['http://localhost:9200'])
            index_name: Nom de l'index Elasticsearch
        """
        if hosts is None:
            hosts = ['http://localhost:9200']
        
        self.index_name = index_name
        
        try:
            self.es = Elasticsearch(hosts)
            # Vérifier la connexion
            if self.es.ping():
                logger.info("✅ Connexion à Elasticsearch réussie")
            else:
                logger.error("❌ Échec de connexion à Elasticsearch")
        except Exception as e:
            logger.error(f"❌ Erreur de connexion à Elasticsearch: {e}")
            self.es = None
    
    def log_metrics(self, 
                   run_id: str,
                   metrics: Dict[str, float],
                   params: Optional[Dict[str, Any]] = None,
                   tags: Optional[Dict[str, str]] = None) -> bool:
        """
        Envoie les métriques MLflow vers Elasticsearch
        
        Args:
            run_id: ID de l'exécution MLflow
            metrics: Dictionnaire des métriques
            params: Dictionnaire des paramètres (optionnel)
            tags: Dictionnaire des tags (optionnel)
            
        Returns:
            bool: True si l'envoi a réussi, False sinon
        """
        if self.es is None:
            logger.warning("⚠️ Elasticsearch non connecté, impossible d'envoyer les logs")
            return False
        
        try:
            document = {
                'run_id': run_id,
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics,
                'params': params or {},
                'tags': tags or {}
            }
            
            response = self.es.index(
                index=self.index_name,
                document=document
            )
            
            logger.info(f"✅ Métriques envoyées à Elasticsearch (ID: {response['_id']})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'envoi vers Elasticsearch: {e}")
            return False
    
    def log_model_info(self,
                      run_id: str,
                      model_name: str,
                      model_version: str,
                      metrics: Dict[str, float],
                      additional_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Envoie les informations du modèle vers Elasticsearch
        
        Args:
            run_id: ID de l'exécution MLflow
            model_name: Nom du modèle
            model_version: Version du modèle
            metrics: Métriques du modèle
            additional_info: Informations supplémentaires
            
        Returns:
            bool: True si l'envoi a réussi
        """
        if self.es is None:
            return False
        
        try:
            document = {
                'run_id': run_id,
                'model_name': model_name,
                'model_version': model_version,
                'timestamp': datetime.utcnow().isoformat(),
                'metrics': metrics,
                'additional_info': additional_info or {}
            }
            
            response = self.es.index(
                index=f"{self.index_name}-models",
                document=document
            )
            
            logger.info(f"✅ Informations du modèle envoyées (ID: {response['_id']})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'envoi des infos du modèle: {e}")
            return False
    
    def log_training_event(self,
                          run_id: str,
                          event_type: str,
                          message: str,
                          additional_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Envoie un événement d'entraînement vers Elasticsearch
        
        Args:
            run_id: ID de l'exécution MLflow
            event_type: Type d'événement (start, end, error, etc.)
            message: Message de l'événement
            additional_data: Données supplémentaires
            
        Returns:
            bool: True si l'envoi a réussi
        """
        if self.es is None:
            return False
        
        try:
            document = {
                'run_id': run_id,
                'event_type': event_type,
                'message': message,
                'timestamp': datetime.utcnow().isoformat(),
                'data': additional_data or {}
            }
            
            response = self.es.index(
                index=f"{self.index_name}-events",
                document=document
            )
            
            logger.info(f"✅ Événement '{event_type}' enregistré")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'envoi de l'événement: {e}")
            return False
    
    def close(self):
        """Ferme la connexion Elasticsearch"""
        if self.es:
            self.es.close()
            logger.info("🔌 Connexion Elasticsearch fermée")

