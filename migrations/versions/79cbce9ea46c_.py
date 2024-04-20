"""empty message

Revision ID: 79cbce9ea46c
Revises: 
Create Date: 2024-04-20 13:35:26.231279

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '79cbce9ea46c'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('hw_db',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('input', sa.String(length=200), nullable=False),
    sa.Column('output', sa.String(length=200), nullable=False),
    sa.Column('create_date', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('jt_db',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('input', sa.String(length=200), nullable=False),
    sa.Column('output', sa.String(length=200), nullable=False),
    sa.Column('create_date', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('na_db',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('input', sa.String(length=200), nullable=False),
    sa.Column('output', sa.String(length=200), nullable=False),
    sa.Column('create_date', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('sm_db',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('input', sa.String(length=200), nullable=False),
    sa.Column('output', sa.String(length=200), nullable=False),
    sa.Column('create_date', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('sm_db')
    op.drop_table('na_db')
    op.drop_table('jt_db')
    op.drop_table('hw_db')
    # ### end Alembic commands ###