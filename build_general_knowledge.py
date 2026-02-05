import sys
import os
import chromadb
from sentence_transformers import SentenceTransformer
import json

sys.stderr = open(os.devnull, 'w')

class GeneralKnowledgeBuilder:
    
    def __init__(self, db_path="./test_memory_db"):
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_collection("test_cases")
        self.embedder = SentenceTransformer('/models/iic/nlp_corom_sentence-embedding_chinese-base')
    
    def add_general_scenes(self):
        
        general_knowledge = self._get_general_knowledge()
        
        print(f"Adding {len(general_knowledge)} general scenarios...")
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for item in general_knowledge:
            doc_text = f"Scene: {item['scene']}\nDescription: {item['description']}\nFeatures: {', '.join(item['features'])}"
            
            embedding = self.embedder.encode(doc_text).tolist()
            
            ids.append(item['id'])
            documents.append(doc_text)
            embeddings.append(embedding)
            metadatas.append({
                'scene': item['scene'],
                'standard_output': json.dumps(item['standard_output'], ensure_ascii=False)
            })
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"Done! Total records: {self.collection.count()}")
    
    def _get_general_knowledge(self):
        return [
            {
                "id": "general_leaving_home",
                "scene": "general_daily",
                "description": "Person leaving home, checking surroundings",
                "features": ["home environment", "door", "leaving action", "checking items"],
                "standard_output": {
                    "scene": "home",
                    "action": "leaving",
                    "need_reminder": True,
                    "reminder_message": "Check if you have wallet, phone, and keys before leaving",
                    "urgency": "medium"
                }
            },
            {
                "id": "general_eating_restaurant",
                "scene": "general_daily",
                "description": "Person eating at restaurant, finishing meal",
                "features": ["restaurant", "eating", "table", "finishing meal"],
                "standard_output": {
                    "scene": "restaurant",
                    "action": "dining",
                    "need_reminder": False,
                    "reminder_message": ""
                }
            },
            {
                "id": "general_stranger_approach",
                "scene": "general_safety",
                "description": "Stranger approaching at door or street, sales or suspicious behavior",
                "features": ["stranger", "door", "street", "approaching", "sales"],
                "standard_output": {
                    "scene": "safety_concern",
                    "action": "stranger_interaction",
                    "need_reminder": True,
                    "reminder_message": "Be cautious with strangers. Verify identity before engaging",
                    "urgency": "high"
                }
            },
            {
                "id": "general_crossing_street",
                "scene": "general_safety",
                "description": "Person crossing street or near traffic",
                "features": ["outdoor", "street", "traffic", "crossing", "vehicles"],
                "standard_output": {
                    "scene": "outdoor_traffic",
                    "action": "crossing_street",
                    "need_reminder": True,
                    "reminder_message": "Watch for traffic. Look both ways before crossing",
                    "urgency": "high"
                }
            },
            {
                "id": "general_kitchen_cooking",
                "scene": "general_daily",
                "description": "Person cooking in kitchen, using stove or appliances",
                "features": ["kitchen", "cooking", "stove", "appliances", "food"],
                "standard_output": {
                    "scene": "kitchen",
                    "action": "cooking",
                    "need_reminder": True,
                    "reminder_message": "Remember to turn off stove when done cooking",
                    "urgency": "medium"
                }
            },
            {
                "id": "general_medication_time",
                "scene": "general_health",
                "description": "Person at home during typical medication time",
                "features": ["home", "morning", "evening", "routine", "pills"],
                "standard_output": {
                    "scene": "home",
                    "action": "daily_routine",
                    "need_reminder": True,
                    "reminder_message": "Time to take your medication",
                    "urgency": "high"
                }
            },
            {
                "id": "general_shopping_payment",
                "scene": "general_daily",
                "description": "Person shopping at store, at checkout counter",
                "features": ["store", "shopping", "checkout", "payment", "cashier"],
                "standard_output": {
                    "scene": "store",
                    "action": "shopping",
                    "need_reminder": False,
                    "reminder_message": ""
                }
            },
            {
                "id": "general_exercise_outdoor",
                "scene": "general_health",
                "description": "Person exercising outdoors, running or walking",
                "features": ["outdoor", "exercise", "running", "walking", "park"],
                "standard_output": {
                    "scene": "outdoor_exercise",
                    "action": "exercising",
                    "need_reminder": False,
                    "reminder_message": ""
                }
            },
            {
                "id": "general_phone_distracted",
                "scene": "general_safety",
                "description": "Person walking while looking at phone",
                "features": ["walking", "phone", "distracted", "outdoor", "not watching"],
                "standard_output": {
                    "scene": "outdoor",
                    "action": "distracted_walking",
                    "need_reminder": True,
                    "reminder_message": "Watch your surroundings while walking",
                    "urgency": "medium"
                }
            },
            {
                "id": "general_meeting_people",
                "scene": "general_social",
                "description": "Person meeting others in social setting",
                "features": ["people", "greeting", "social", "conversation"],
                "standard_output": {
                    "scene": "social_interaction",
                    "action": "meeting_people",
                    "need_reminder": False,
                    "reminder_message": ""
                }
            },
            {
                "id": "general_stairs_elderly",
                "scene": "general_safety",
                "description": "Elderly person using stairs",
                "features": ["stairs", "elderly", "climbing", "indoor", "caution"],
                "standard_output": {
                    "scene": "indoor_stairs",
                    "action": "using_stairs",
                    "need_reminder": True,
                    "reminder_message": "Hold the handrail and take your time on stairs",
                    "urgency": "medium"
                }
            },
            {
                "id": "general_rain_outdoor",
                "scene": "general_daily",
                "description": "Person going outside in rainy weather",
                "features": ["outdoor", "rain", "wet", "umbrella"],
                "standard_output": {
                    "scene": "outdoor_rainy",
                    "action": "going_out",
                    "need_reminder": True,
                    "reminder_message": "Don't forget your umbrella. Watch for slippery surfaces",
                    "urgency": "medium"
                }
            },
            {
                "id": "general_public_transport",
                "scene": "general_daily",
                "description": "Person using bus or subway",
                "features": ["bus", "subway", "public transport", "commuting"],
                "standard_output": {
                    "scene": "public_transport",
                    "action": "commuting",
                    "need_reminder": False,
                    "reminder_message": ""
                }
            },
            {
                "id": "general_atm_banking",
                "scene": "general_daily",
                "description": "Person using ATM or at bank",
                "features": ["ATM", "bank", "money", "transaction"],
                "standard_output": {
                    "scene": "banking",
                    "action": "transaction",
                    "need_reminder": True,
                    "reminder_message": "Keep your PIN private and take your card when done",
                    "urgency": "medium"
                }
            },
            {
                "id": "general_parking_car",
                "scene": "general_daily",
                "description": "Person parking car or getting out of vehicle",
                "features": ["car", "parking", "vehicle", "outdoor"],
                "standard_output": {
                    "scene": "parking_area",
                    "action": "parking",
                    "need_reminder": True,
                    "reminder_message": "Check if you locked the car and have all belongings",
                    "urgency": "medium"
                }
            }
        ]

if __name__ == "__main__":
    sys.stderr = sys.__stderr__
    builder = GeneralKnowledgeBuilder()
    
    print(f"Current records: {builder.collection.count()}")
    builder.add_general_scenes()
    print("General knowledge added successfully!")
