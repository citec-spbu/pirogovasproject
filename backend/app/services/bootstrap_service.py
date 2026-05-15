from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.report import Report
from app.models.report_templates import ReportTemplate
from app.models.clinical_protocols import ClinicalProtocol
from app.models.llm_calls import LLMCall
from app.models.user import User, Organization

from app.core.config import get_settings
from app.core.enum.role import UserRole
from app.core.security import get_password_hash
from app.core.database import engine, Base, AsyncSessionLocal

async def create_tables() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def bootstrap_first_admin(db: AsyncSession) -> None:
    admin_exists_result = await db.execute(
        select(User).where(User.role == UserRole.ADMIN).limit(1)
    )
    admin_exists = admin_exists_result.scalar_one_or_none()

    if admin_exists:
        return
    
    organization_exists_result = await db.execute(
        select(Organization).where(Organization.name == get_settings().FIRST_ORGANIZATION_NAME)
    )
    organization = organization_exists_result.scalar_one_or_none()

    if not organization:
        organization = Organization(
            name=get_settings().FIRST_ORGANIZATION_NAME
        )
        db.add(organization)
        await db.commit()
        await db.refresh(organization)

    first_admin = User(
        login=get_settings().FIRST_ADMIN_LOGIN,
        hashed_password=get_password_hash(get_settings().FIRST_ADMIN_PASSWORD),
        role=UserRole.ADMIN,
        organization_name=organization.name,
        name = get_settings().FIRST_ADMIN_NAME,
        surname = get_settings().FIRST_ADMIN_SURNAME,
        patronymic = get_settings().FIRST_ADMIN_PATRONYMIC,
        date_of_birth = get_settings().FIRST_ADMIN_DATE_OF_BIRTH,
        is_active = True
    )

    db.add(first_admin)
    await db.commit()

async def bootstrap() -> None:
    await create_tables()

    async with AsyncSessionLocal() as db:
        await bootstrap_first_admin(db)

