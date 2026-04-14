from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.config import get_config_state, update_config, CONFIG_METADATA
import app.config as config

router = APIRouter(prefix="/api/settings", tags=["settings"])

class SettingsUpdate(BaseModel):
    key: str
    value: str

@router.get("")
async def get_settings():
    """Get all settings and their metadata."""
    try:
        return get_config_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("")
async def update_setting(update: SettingsUpdate):
    """Update a specific setting."""
    if update.key not in CONFIG_METADATA:
        raise HTTPException(status_code=400, detail=f"Invalid setting key: {update.key}")
    
    try:
        old_value = getattr(config, update.key, None)
        update_config(update.key, update.value)
        
        # Side effects for core models
        if update.value != old_value:
            if update.key == "LLM_MODEL":
                from app.services.llm import get_llm_client
                get_llm_client().set_model(update.value)
            
            elif update.key == "EMBEDDING_MODEL":
                from app.services.embeddings import get_embedding_service
                from app.services.vector_store import get_vector_store
                es = get_embedding_service()
                # Determine mode
                mode = "sentence-transformers" if "MiniLM" in update.value else "ollama"
                result = await es.switch_model(update.value, mode=mode)
                
                if result.get("dim_changed"):
                    vs = get_vector_store()
                    vs.initialize(dim=result["dim"])
                    vs.save()

        return {"success": True, "key": update.key, "value": update.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
